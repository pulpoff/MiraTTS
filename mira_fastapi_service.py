import logging
import time
import os
import tempfile
import subprocess
import gc
from contextlib import contextmanager
from pathlib import Path
import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import Optional, Dict, List
import warnings
import numpy as np
import scipy.io.wavfile as wav

from mira.streaming_model import MiraTTSStreaming

warnings.filterwarnings('ignore')

# --- Configuration ---
FINAL_SAMPLE_RATE = 16000
TEMPO_FACTOR = 1.1
MASTER_VOLUME_GAIN = 0.8
DEFAULT_SPEED = 1.0  # MiraTTS doesn't have speed parameter like Kokoro
MIRA_OUTPUT_SAMPLE_RATE = 48000  # MiraTTS generates 48kHz audio
STREAMING_CHUNK_SIZE = 50  # Number of tokens to accumulate before decoding (lower = faster, higher = more efficient)

# --- Voice Directory (Reference Audio Files) ---
VOICES_DIR = Path("/ref")
if not VOICES_DIR.exists():
    VOICES_DIR = Path("./ref")

print(f"Using voice directory: {VOICES_DIR.absolute()}")

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Discover Voice Files ---
def discover_voices() -> Dict:
    """Scan directory for reference audio files (.wav, .mp3, .ogg) and their text transcripts"""
    voices = {}

    print(f"Scanning for reference audio files in {VOICES_DIR}...")

    # Supported audio formats
    audio_extensions = ['*.wav', '*.mp3', '*.ogg', '*.flac', '*.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(list(VOICES_DIR.glob(ext)))

    if not audio_files:
        print(f"❌ No audio files found in {VOICES_DIR}")
        return {}

    for voice_path in audio_files:
        voice_id = voice_path.stem

        # Look for corresponding text file
        text_path = VOICES_DIR / f"{voice_id}.txt"
        reference_text = None
        has_text = False

        if text_path.exists():
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    reference_text = f.read().strip()
                has_text = True
            except Exception as e:
                print(f"⚠️  Warning: Could not read {text_path}: {e}")

        voices[voice_id] = {
            'name': voice_id.replace('_', ' ').title(),
            'path': str(voice_path),
            'format': voice_path.suffix[1:].upper(),
            'file_size_mb': round(voice_path.stat().st_size / (1024 * 1024), 2),
            'reference_text': reference_text,
            'has_reference_text': has_text
        }

    return dict(sorted(voices.items()))

# --- Initialize MiraTTS ---
AVAILABLE_VOICES = discover_voices()

print("\n" + "="*60)
print("Initializing MiraTTS Pipeline")
print("="*60)

if not AVAILABLE_VOICES:
    print("❌ WARNING: No reference audio files found!")
    print(f"Voice directory: {VOICES_DIR}")
    print("Service will start but TTS will fail without reference audio files.")

# Initialize MiraTTS
try:
    # Use streaming-enabled MiraTTS
    MIRA_TTS = MiraTTSStreaming(
        model_dir="YatharthS/MiraTTS",
        tp=1,  # Tensor parallelism (increase if multiple GPUs)
        enable_prefix_caching=True,
        cache_max_entry_count=0.5  # Increased for better performance
    )
    print(f"✓ MiraTTS initialized")
    print(f"  Model: YatharthS/MiraTTS")
    print(f"  Output sample rate: {MIRA_OUTPUT_SAMPLE_RATE}Hz")
    print(f"  Device: {device}")

    if AVAILABLE_VOICES:
        print(f"\nFound {len(AVAILABLE_VOICES)} reference voice(s):")
        for voice_id in sorted(AVAILABLE_VOICES.keys()):
            voice_info = AVAILABLE_VOICES[voice_id]
            print(f"  • {voice_id}: {voice_info['name']} ({voice_info['format']}, {voice_info['file_size_mb']}MB)")

except Exception as e:
    print(f"✗ Failed to initialize MiraTTS: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Default voice
DEFAULT_VOICE = list(AVAILABLE_VOICES.keys())[0] if AVAILABLE_VOICES else None

if DEFAULT_VOICE:
    print(f"\nDefault voice: {DEFAULT_VOICE}")
print("="*60)

# --- Voice Context Cache ---
# Cache encoded context tokens to avoid re-encoding the same reference audio
voice_context_cache = {}

def get_voice_context(voice_id: str):
    """Get or cache voice context tokens"""
    if voice_id not in voice_context_cache:
        if voice_id not in AVAILABLE_VOICES:
            raise ValueError(f"Voice '{voice_id}' not found")

        voice_path = AVAILABLE_VOICES[voice_id]['path']
        print(f"Encoding reference audio: {voice_id}")
        context_tokens = MIRA_TTS.encode_audio(voice_path)
        voice_context_cache[voice_id] = context_tokens
        print(f"✓ Cached context tokens for {voice_id}")

    return voice_context_cache[voice_id]

# --- Helper Functions ---
def validate_voice(voice_id: str) -> bool:
    """Check if voice ID is valid"""
    return voice_id in AVAILABLE_VOICES

def get_voice_info(voice_id: str) -> Dict:
    """Get information about a specific voice"""
    if voice_id not in AVAILABLE_VOICES:
        return None
    info = AVAILABLE_VOICES[voice_id].copy()
    info['id'] = voice_id
    info['usage_count'] = voice_usage_stats.get(voice_id, 0)
    info['cached'] = voice_id in voice_context_cache
    return info

# --- FastAPI Setup ---
app = FastAPI(
    title="MiraTTS FastAPI Server",
    description=f"High-performance Text-to-Speech API using MiraTTS with {len(AVAILABLE_VOICES)} reference voices. Real chunked streaming via LMDeploy.",
    version="2.0.0"  # Real chunked streaming
)

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.error').setLevel(logging.ERROR)
logging.getLogger('uvicorn.access').setLevel(logging.ERROR)

request_count = 0
voice_usage_stats = {voice: 0 for voice in AVAILABLE_VOICES.keys()}

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    input: str
    voice: Optional[str] = DEFAULT_VOICE
    speed: Optional[float] = DEFAULT_SPEED  # For compatibility, but MiraTTS doesn't use this
    tempo: Optional[float] = TEMPO_FACTOR
    volume: Optional[float] = MASTER_VOLUME_GAIN

    class Config:
        json_schema_extra = {
            "example": {
                "input": "Hello, this is a test of the MiraTTS system.",
                "voice": DEFAULT_VOICE,
                "speed": 1.0,
                "tempo": 1.1,
                "volume": 0.8
            }
        }

# --- Audio Processing Functions ---
@contextmanager
def managed_ffmpeg(command):
    """Context manager for FFmpeg process lifecycle"""
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True
        )
        yield process
    finally:
        if process:
            for pipe in [process.stdin, process.stdout, process.stderr]:
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except:
                        pass
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    try:
                        process.kill()
                        process.wait()
                    except:
                        pass

def process_audio_chunk_with_ffmpeg(wav_file_path, output_sample_rate=16000, tempo_factor=1.0, volume_gain=0.8):
    """Process audio file with FFmpeg and return raw PCM data."""
    try:
        # Check if file exists (might be deleted if client disconnected)
        if not os.path.exists(wav_file_path):
            return b''

        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', wav_file_path,
            '-ar', str(output_sample_rate),
            '-ac', '1',
            '-filter:a', f'volume={volume_gain},atempo={tempo_factor}',
            '-f', 's16le',
            '-',
            '-loglevel', 'error'
        ]

        with managed_ffmpeg(ffmpeg_command) as process:
            stdout, stderr = process.communicate(timeout=30)
            if process.returncode != 0:
                if os.path.exists(wav_file_path):
                    raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
                else:
                    return b''
            return stdout

    except FileNotFoundError:
        return b''
    except Exception as e:
        if os.path.exists(wav_file_path):
            print(f"FFmpeg error: {e}")
            raise
        else:
            return b''

def generate_mira_audio(text: str, voice: str):
    """Generate audio using MiraTTS with the specified reference voice."""
    if not validate_voice(voice):
        raise ValueError(f"Voice '{voice}' is not available.")

    voice_info = AVAILABLE_VOICES[voice]
    reference_text = voice_info.get('reference_text')

    if reference_text:
        print(f"Generating audio with voice: {voice} ({voice_info['name']}) [with reference text]")
    else:
        print(f"Generating audio with voice: {voice} ({voice_info['name']}) [no reference text]")

    # Track voice usage
    global voice_usage_stats
    voice_usage_stats[voice] = voice_usage_stats.get(voice, 0) + 1

    # Get context tokens (cached)
    context_tokens = get_voice_context(voice)

    # Generate audio with reference text
    audio_tensor = MIRA_TTS.generate(text, context_tokens, reference_text=reference_text)

    # Convert tensor to numpy array
    if isinstance(audio_tensor, torch.Tensor):
        audio_numpy = audio_tensor.cpu().numpy()
    else:
        audio_numpy = audio_tensor

    # Ensure it's 1D
    if audio_numpy.ndim > 1:
        audio_numpy = audio_numpy.flatten()

    return audio_numpy

# --- Non-Streaming Helper ---
def run_non_streaming_inference(prompt: str, voice: str,
                               tempo: float = TEMPO_FACTOR,
                               volume: float = MASTER_VOLUME_GAIN):
    """Run the entire blocking workflow for non-streaming TTS."""
    temp_output_path = None
    try:
        print(f"Converting text: '{prompt[:50]}...' with voice: {voice}")

        start_time = time.time()

        audio = generate_mira_audio(prompt, voice)

        if audio.size == 0:
            raise RuntimeError("MiraTTS synthesis failed, returned no audio.")

        inference_end_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_output_path = f.name

        # MiraTTS outputs 48kHz audio
        wav.write(temp_output_path, MIRA_OUTPUT_SAMPLE_RATE, audio)

        raw_pcm = process_audio_chunk_with_ffmpeg(temp_output_path, FINAL_SAMPLE_RATE, tempo, volume)

        end_time = time.time()

        print(f"Time taken (Inference only): {inference_end_time - start_time:.2f} seconds")
        print(f"Time taken (I/O & Conversion): {end_time - inference_end_time:.2f} seconds")
        print(f"Time taken (Total): {end_time - start_time:.2f} seconds")

        return raw_pcm

    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- FastAPI Endpoints ---

@app.post('/v1/audio/speech')
async def generate_audio_endpoint(request: TTSRequest):
    """Non-streaming endpoint - generates complete audio file"""
    global request_count
    request_count += 1

    try:
        if not request.voice:
            raise HTTPException(
                status_code=400,
                detail="Voice parameter is required. Use /voices endpoint to see available voices."
            )

        if not validate_voice(request.voice):
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' is not available. Use /voices endpoint to see available voices."
            )

        raw_pcm = await asyncio.to_thread(
            run_non_streaming_inference,
            request.input,
            request.voice,
            request.tempo or TEMPO_FACTOR,
            request.volume or MASTER_VOLUME_GAIN
        )

        voice_info = AVAILABLE_VOICES[request.voice]
        return Response(
            content=raw_pcm,
            media_type="application/octet-stream",
            headers={
                "X-Voice-Used": request.voice,
                "X-Voice-Name": voice_info['name'],
                "X-Voice-Format": voice_info['format']
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Async Helper for Generators ---
async def asyncio_wrap_generator(sync_generator):
    """Wrap a synchronous generator to work with async for loops"""
    for item in sync_generator:
        yield item
        # Allow other async tasks to run
        await asyncio.sleep(0)

async def async_streaming_generator(prompt: str, voice: str,
                                   tempo: float = TEMPO_FACTOR,
                                   volume: float = MASTER_VOLUME_GAIN):
    """
    Async generator for streaming TTS output.

    Uses real chunked streaming from LMDeploy - generates and yields audio
    chunks as tokens are produced, providing minimal latency.
    """
    global request_count
    request_count += 1

    if not validate_voice(voice):
        raise ValueError(f"Voice '{voice}' is not available.")

    voice_info = AVAILABLE_VOICES[voice]
    reference_text = voice_info.get('reference_text')

    if reference_text:
        print(f"Streaming TTS with voice: {voice} ({voice_info['name']}) [Real chunked streaming, with reference text]")
    else:
        print(f"Streaming TTS with voice: {voice} ({voice_info['name']}) [Real chunked streaming, no reference text]")

    chunk_count = 0
    total_bytes = 0
    inference_start = time.time()
    temp_files = []

    try:
        # Get cached context tokens
        context_tokens = get_voice_context(voice)

        # Generate audio chunks from MiraTTS using real token-level streaming
        async for audio_chunk in asyncio_wrap_generator(
            MIRA_TTS.stream_generate(prompt, context_tokens, chunk_size=STREAMING_CHUNK_SIZE, reference_text=reference_text)
        ):
            i = chunk_count
            if audio_chunk is None or audio_chunk.size == 0:
                continue

            chunk_count += 1
            chunk_start = time.time()

            # Initialize variables for safe cleanup
            temp_wav_path = None
            raw_pcm = None

            try:
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    temp_wav_path = f.name
                temp_files.append(temp_wav_path)

                # MiraTTS outputs 48kHz audio
                await asyncio.to_thread(wav.write, temp_wav_path, MIRA_OUTPUT_SAMPLE_RATE, audio_chunk)

                # Process with FFmpeg
                raw_pcm = await asyncio.to_thread(
                    process_audio_chunk_with_ffmpeg,
                    temp_wav_path,
                    FINAL_SAMPLE_RATE,
                    tempo,
                    volume
                )

                # Ensure even byte alignment for s16le format
                pcm_len = len(raw_pcm)
                aligned_len = (pcm_len // 2) * 2
                if aligned_len < pcm_len:
                    raw_pcm = raw_pcm[:aligned_len]

                if len(raw_pcm) > 0:
                    chunk_bytes = len(raw_pcm)
                    total_bytes += chunk_bytes

                    chunk_end = time.time()

                    if i == 0:
                        first_chunk_latency = chunk_end - inference_start
                        print(f"✓ First chunk latency: {first_chunk_latency:.3f}s")

                    print(f"  Chunk {chunk_count} processed in {chunk_end - chunk_start:.3f}s ({chunk_bytes:,} bytes)")

                    yield raw_pcm

            except asyncio.CancelledError:
                # Client disconnected - clean up and break
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                        if temp_wav_path in temp_files:
                            temp_files.remove(temp_wav_path)
                    except:
                        pass

                # Log disconnect
                if chunk_count == 0:
                    print(f"✗ Client disconnected before first chunk")
                elif total_bytes == 0:
                    print(f"✗ Client disconnected during chunk {chunk_count} processing")
                else:
                    print(f"✗ Client disconnected after {chunk_count} chunk(s), {total_bytes:,} bytes sent")

                break

            finally:
                # Clean up temp file after successful processing
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                        if temp_wav_path in temp_files:
                            temp_files.remove(temp_wav_path)
                    except:
                        pass

                # Safe cleanup of variables
                if raw_pcm is not None:
                    del raw_pcm
                if audio_chunk is not None:
                    del audio_chunk

            # Periodic garbage collection
            if chunk_count % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        inference_end = time.time()

        # Only log completion if we actually completed
        if chunk_count > 0 and total_bytes > 0:
            print(f"✓ Streaming complete. Total chunks: {chunk_count}, Total bytes: {total_bytes:,}")
            print(f"Inference time: {inference_end - inference_start:.2f}s")

    except Exception as e:
        print(f"✗ Error in streaming generator after {chunk_count} chunks: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up any remaining temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post('/v1/audio/speech-stream')
async def generate_audio_stream_endpoint(request: TTSRequest):
    """Streaming endpoint - generates audio in chunks with minimal latency."""
    try:
        if not request.voice:
            raise HTTPException(
                status_code=400,
                detail="Voice parameter is required. Use /voices endpoint to see available voices."
            )

        if not validate_voice(request.voice):
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' is not available. Use /voices endpoint to see available voices."
            )

        voice_info = AVAILABLE_VOICES[request.voice]
        return StreamingResponse(
            async_streaming_generator(
                request.input,
                request.voice,
                request.tempo or TEMPO_FACTOR,
                request.volume or MASTER_VOLUME_GAIN
            ),
            media_type="application/octet-stream",
            headers={
                "X-Voice-Used": request.voice,
                "X-Voice-Name": voice_info['name'],
                "X-Voice-Format": voice_info['format']
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/voices')
async def list_voices():
    """List all available reference voices."""
    voices_with_info = {}

    for voice_id, info in AVAILABLE_VOICES.items():
        voices_with_info[voice_id] = {
            **info,
            'id': voice_id,
            'usage_count': voice_usage_stats.get(voice_id, 0),
            'cached': voice_id in voice_context_cache
        }

    return {
        'total': len(voices_with_info),
        'default_voice': DEFAULT_VOICE,
        'voices_directory': str(VOICES_DIR),
        'voices': voices_with_info
    }

@app.get('/voices/{voice_id}')
async def get_voice_details(voice_id: str):
    """Get detailed information about a specific voice."""
    if not validate_voice(voice_id):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Use /voices endpoint to see available voices."
        )

    return get_voice_info(voice_id)

@app.get('/voices/refresh')
async def refresh_voices():
    """Refresh the list of available voices from the directory."""
    global AVAILABLE_VOICES, DEFAULT_VOICE, voice_usage_stats, voice_context_cache

    old_count = len(AVAILABLE_VOICES)
    AVAILABLE_VOICES = discover_voices()

    if AVAILABLE_VOICES:
        if DEFAULT_VOICE not in AVAILABLE_VOICES:
            DEFAULT_VOICE = list(AVAILABLE_VOICES.keys())[0]
    else:
        DEFAULT_VOICE = None

    # Update stats
    new_stats = {}
    for voice in AVAILABLE_VOICES:
        new_stats[voice] = voice_usage_stats.get(voice, 0)
    voice_usage_stats = new_stats

    # Clear context cache for removed voices
    removed_voices = set(voice_context_cache.keys()) - set(AVAILABLE_VOICES.keys())
    for voice in removed_voices:
        del voice_context_cache[voice]

    return {
        'status': 'refreshed',
        'previous_count': old_count,
        'current_count': len(AVAILABLE_VOICES),
        'default_voice': DEFAULT_VOICE,
        'voices_directory': str(VOICES_DIR),
        'cached_contexts_cleared': len(removed_voices)
    }

@app.post('/voices/clear-cache')
async def clear_voice_cache():
    """Clear the voice context token cache to free memory."""
    global voice_context_cache

    cache_size = len(voice_context_cache)
    voice_context_cache = {}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'status': 'cache_cleared',
        'voices_cleared': cache_size,
        'message': 'Voice context cache cleared. Next TTS request will re-encode reference audio.'
    }

@app.get('/stats')
async def get_service_stats():
    """Get service statistics including voice usage."""
    total_requests = request_count
    voice_stats = []

    for voice_id, count in sorted(voice_usage_stats.items(), key=lambda x: x[1], reverse=True):
        if voice_id in AVAILABLE_VOICES:
            info = AVAILABLE_VOICES[voice_id].copy()
            info['id'] = voice_id
            info['usage_count'] = count
            info['usage_percentage'] = round((count / total_requests) * 100, 2) if total_requests > 0 else 0
            info['cached'] = voice_id in voice_context_cache
            voice_stats.append(info)

    return {
        'total_requests': total_requests,
        'available_voices': len(AVAILABLE_VOICES),
        'cached_voices': len(voice_context_cache),
        'default_voice': DEFAULT_VOICE,
        'voices_directory': str(VOICES_DIR),
        'voice_usage': voice_stats,
        'most_used_voice': voice_stats[0] if voice_stats else None
    }

@app.get('/health')
async def health_check():
    """Health check endpoint."""
    try:
        import psutil
        memory = psutil.virtual_memory()

        health = {
            'status': 'healthy' if memory.percent < 90 else 'degraded',
            'memory_percent': memory.percent,
            'requests_processed': request_count,
            'available_voices': len(AVAILABLE_VOICES),
            'cached_voices': len(voice_context_cache),
            'default_voice': DEFAULT_VOICE,
            'voices_directory': str(VOICES_DIR),
            'service': 'MiraTTS FastAPI Server v2.0.0 (Real Chunked Streaming)'
        }

        if torch.cuda.is_available():
            health['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated() / (1024**2), 2)
            health['gpu_memory_cached_mb'] = round(torch.cuda.memory_reserved() / (1024**2), 2)

        status_code = 503 if health['status'] != 'healthy' else 200
        return JSONResponse(content=health, status_code=status_code)
    except ImportError:
        return JSONResponse(content={
            'status': 'healthy',
            'requests_processed': request_count,
            'available_voices': len(AVAILABLE_VOICES),
            'cached_voices': len(voice_context_cache),
            'default_voice': DEFAULT_VOICE,
            'voices_directory': str(VOICES_DIR)
        })

@app.get('/')
async def root():
    """Root endpoint with service information."""
    return {
        'service': 'MiraTTS FastAPI Server',
        'version': '2.0.0',
        'description': 'High-performance Text-to-Speech service using MiraTTS with streaming support',
        'model': 'YatharthS/MiraTTS',
        'total_voices': len(AVAILABLE_VOICES),
        'default_voice': DEFAULT_VOICE,
        'voices_directory': str(VOICES_DIR),
        'output_quality': '48kHz high-quality audio (downsampled to 16kHz for output)',
        'features': [
            'Voice cloning via reference audio files',
            'Real chunked streaming (token-level with LMDeploy)',
            'Low latency inference (~100ms)',
            'High quality 48kHz audio generation',
            'Voice context caching for performance',
            'Compatible with Kokoro TTS API'
        ],
        'endpoints': {
            '/v1/audio/speech': 'Generate TTS audio (non-streaming)',
            '/v1/audio/speech-stream': 'Generate TTS audio (streaming)',
            '/voices': 'List all available reference voices',
            '/voices/{voice_id}': 'Get voice details',
            '/voices/refresh': 'Refresh voice list from directory',
            '/voices/clear-cache': 'Clear voice context cache',
            '/stats': 'Get service statistics',
            '/health': 'Health check',
            '/docs': 'API documentation (Swagger UI)'
        }
    }

if __name__ == "__main__":
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ FFmpeg found")
    except FileNotFoundError:
        print("FATAL ERROR: FFmpeg is required but not found in the environment PATH.")
        exit(1)

    # Check if we have any voices
    if not AVAILABLE_VOICES:
        print("\n" + "!" * 60)
        print("WARNING: No reference audio files found in directory!")
        print(f"Directory: {VOICES_DIR}")
        print("\nPlease place reference audio files (.wav, .mp3, .ogg, etc.) in the directory.")
        print("These files will be used as voice references for TTS generation.")
        print("!" * 60)

    print("\n" + "="*60)
    print("MIRATTS FASTAPI SERVER v2.0.0 (Real Chunked Streaming)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: YatharthS/MiraTTS")
    print(f"Available voices: {len(AVAILABLE_VOICES)} reference audio file(s)")
    if DEFAULT_VOICE:
        print(f"Default voice: {DEFAULT_VOICE}")
    print(f"Voice directory: {VOICES_DIR}")
    print(f"Output sample rate: {MIRA_OUTPUT_SAMPLE_RATE}Hz -> {FINAL_SAMPLE_RATE}Hz")
    print(f"Streaming: Real chunked streaming (token-level, chunk_size={STREAMING_CHUNK_SIZE})")
    print(f"Listening on: http://0.0.0.0:5100")
    print(f"Docs available at: http://0.0.0.0:5100/docs")
    print("="*60)
    print("\n*** PRODUCTION DEPLOYMENT ***")
    print("For production, use Gunicorn with multiple workers:")
    print("  gunicorn -w 4 -k uvicorn.workers.UvicornWorker mira_fastapi_service:app \\")
    print("    --bind 0.0.0.0:5100 --timeout 120 --keep-alive 5")
    print("***\n")

    # Run with Uvicorn for development
    uvicorn.run(app, host='0.0.0.0', port=5100, timeout_keep_alive=30)
