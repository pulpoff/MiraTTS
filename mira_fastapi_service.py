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
import soundfile as sf

from mira.streaming_model import MiraTTSStreaming

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
logging.getLogger('lmdeploy').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure HF cache to persist downloads
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/tmp/huggingface_cache')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', '/tmp/huggingface_cache')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

FINAL_SAMPLE_RATE = 16000
TEMPO_FACTOR = 1.1
MASTER_VOLUME_GAIN = 0.8
DEFAULT_SPEED = 1.0
MIRA_OUTPUT_SAMPLE_RATE = 48000
STREAMING_CHUNK_SIZE = 150  # Characters per text chunk (MeloTTS-style)

VOICES_DIR = Path("/voices") if Path("/voices").exists() else Path("./voices")
print(f"Using voice directory: {VOICES_DIR.absolute()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def discover_voices() -> Dict:
    """Scan directory for reference audio files and their text transcripts"""
    voices = {}
    print(f"Scanning for reference audio files in {VOICES_DIR}...")

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac', '*.m4a']:
        audio_files.extend(list(VOICES_DIR.glob(ext)))

    if not audio_files:
        print(f"❌ No audio files found in {VOICES_DIR}")
        return {}

    for voice_path in audio_files:
        voice_id = voice_path.stem

        # Validate audio file can be opened
        try:
            with sf.SoundFile(str(voice_path)) as f:
                pass  # Just checking if file can be opened
        except Exception as e:
            print(f"⚠️  Skipping invalid audio file '{voice_id}': {e}")
            continue

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

AVAILABLE_VOICES = discover_voices()
DEFAULT_VOICE = list(AVAILABLE_VOICES.keys())[0] if AVAILABLE_VOICES else None

# Global model instance - will be initialized at startup
MIRA_TTS = None

def initialize_model():
    """Initialize MiraTTS model at startup"""
    global MIRA_TTS
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("Initializing MiraTTS model...")
    MIRA_TTS = MiraTTSStreaming(
        model_dir="YatharthS/MiraTTS",
        tp=1,
        enable_prefix_caching=True,
        cache_max_entry_count=0.5
    )
    print("✓ Model initialized")

    # Pre-cache default voice for low latency
    if DEFAULT_VOICE:
        print(f"Pre-caching default voice: {DEFAULT_VOICE}")
        get_voice_context(DEFAULT_VOICE)
        print(f"✓ Default voice ready")

def get_mira_tts():
    """Get the initialized MiraTTS model"""
    return MIRA_TTS

voice_context_cache = {}

def get_voice_context(voice_id: str):
    if voice_id not in voice_context_cache:
        if voice_id not in AVAILABLE_VOICES:
            raise ValueError(f"Voice '{voice_id}' not found")
        voice_path = AVAILABLE_VOICES[voice_id]['path']
        mira_tts = get_mira_tts()

        try:
            voice_context_cache[voice_id] = mira_tts.encode_audio(voice_path)
        except Exception as e:
            print(f"✗ Failed to encode audio for '{voice_id}': {e}")
            if voice_id != DEFAULT_VOICE:
                print(f"⚠️  Falling back to default voice: {DEFAULT_VOICE}")
                return get_voice_context(DEFAULT_VOICE)
            else:
                raise ValueError(f"Default voice '{DEFAULT_VOICE}' audio file is corrupted or invalid")

    return voice_context_cache[voice_id]

def validate_voice(voice_id: str) -> bool:
    return voice_id in AVAILABLE_VOICES

def get_voice_info(voice_id: str) -> Dict:
    if voice_id not in AVAILABLE_VOICES:
        return None
    info = AVAILABLE_VOICES[voice_id].copy()
    info['id'] = voice_id
    info['usage_count'] = voice_usage_stats.get(voice_id, 0)
    info['cached'] = voice_id in voice_context_cache
    return info
app = FastAPI(
    title="MiraTTS FastAPI Server",
    description=f"High-performance Text-to-Speech API using MiraTTS with {len(AVAILABLE_VOICES)} reference voices. Real chunked streaming via LMDeploy.",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize model and default voice at startup for low latency"""
    initialize_model()

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.error').setLevel(logging.ERROR)
logging.getLogger('uvicorn.access').setLevel(logging.ERROR)

request_count = 0
voice_usage_stats = {voice: 0 for voice in AVAILABLE_VOICES.keys()}

class TTSRequest(BaseModel):
    input: str
    voice: Optional[str] = DEFAULT_VOICE
    speed: Optional[float] = DEFAULT_SPEED
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

@contextmanager
def managed_ffmpeg(command):
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

    try:
        with managed_ffmpeg(ffmpeg_command) as process:
            stdout, stderr = process.communicate(timeout=30)
            if process.returncode != 0 and os.path.exists(wav_file_path):
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
            return stdout
    except (FileNotFoundError, Exception) as e:
        if os.path.exists(wav_file_path):
            print(f"FFmpeg error: {e}")
            raise
        return b''

def generate_mira_audio(text: str, voice: str):
    original_voice = voice
    if not validate_voice(voice):
        voice = DEFAULT_VOICE
        print(f"⚠️  Voice '{original_voice}' not found, falling back to default voice: {voice}")

    voice_info = AVAILABLE_VOICES[voice]
    reference_text = voice_info.get('reference_text')

    global voice_usage_stats
    voice_usage_stats[voice] = voice_usage_stats.get(voice, 0) + 1

    start_time = time.time()
    mira_tts = get_mira_tts()
    context_tokens = get_voice_context(voice)
    audio_tensor = mira_tts.generate(text, context_tokens, reference_text=reference_text)

    audio_numpy = audio_tensor.cpu().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
    audio_numpy = audio_numpy.flatten() if audio_numpy.ndim > 1 else audio_numpy

    total_time = time.time() - start_time
    audio_duration_sec = len(audio_numpy) / MIRA_OUTPUT_SAMPLE_RATE
    print(f"✓ {voice}: Non-streaming Total={total_time:.2f}s Audio={audio_duration_sec:.2f}s Samples={len(audio_numpy):,}")

    return audio_numpy

def run_non_streaming_inference(prompt: str, voice: str,
                               tempo: float = TEMPO_FACTOR,
                               volume: float = MASTER_VOLUME_GAIN):
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

@app.post('/v1/audio/speech')
async def generate_audio_endpoint(request: TTSRequest):
    global request_count
    request_count += 1

    try:
        if not request.voice:
            request.voice = DEFAULT_VOICE
            print(f"⚠️  No voice specified, using default voice: {request.voice}")

        original_voice = request.voice
        if not validate_voice(request.voice):
            request.voice = DEFAULT_VOICE
            print(f"⚠️  Voice '{original_voice}' not found, falling back to default voice: {request.voice}")

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

async def asyncio_wrap_generator(sync_generator):
    for item in sync_generator:
        yield item
        await asyncio.sleep(0)

async def async_streaming_generator(prompt: str, voice: str,
                                   tempo: float = TEMPO_FACTOR,
                                   volume: float = MASTER_VOLUME_GAIN):
    """Async generator for streaming TTS output with real chunked streaming"""
    global request_count
    request_count += 1

    original_voice = voice
    if not validate_voice(voice):
        voice = DEFAULT_VOICE
        print(f"⚠️  Voice '{original_voice}' not found, falling back to default voice: {voice}")

    voice_info = AVAILABLE_VOICES[voice]
    reference_text = voice_info.get('reference_text')

    chunk_count = 0
    total_bytes = 0
    request_start = time.time()
    first_chunk_time = None
    temp_files = []

    try:
        mira_tts = get_mira_tts()
        context_tokens = get_voice_context(voice)

        async for audio_chunk in asyncio_wrap_generator(
            mira_tts.stream_generate(prompt, context_tokens, chunk_size=STREAMING_CHUNK_SIZE, reference_text=reference_text)
        ):
            if audio_chunk is None or audio_chunk.size == 0:
                continue

            chunk_start = time.time()
            temp_wav_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    temp_wav_path = f.name
                temp_files.append(temp_wav_path)

                # Convert torch tensor to numpy array for scipy.io.wavfile.write
                audio_numpy = audio_chunk.cpu().numpy() if hasattr(audio_chunk, 'cpu') else audio_chunk

                # Convert float16/float32 to int16 (standard WAV format)
                if audio_numpy.dtype in [np.float16, np.float32, np.float64]:
                    # Ensure float32 for proper scaling
                    audio_numpy = audio_numpy.astype(np.float32)
                    # Scale to int16 range and convert
                    audio_numpy = (audio_numpy * 32767).astype(np.int16)

                await asyncio.to_thread(wav.write, temp_wav_path, MIRA_OUTPUT_SAMPLE_RATE, audio_numpy)

                raw_pcm = await asyncio.to_thread(
                    process_audio_chunk_with_ffmpeg,
                    temp_wav_path,
                    FINAL_SAMPLE_RATE,
                    tempo,
                    volume
                )

                pcm_len = len(raw_pcm)
                if pcm_len % 2:
                    raw_pcm = raw_pcm[:pcm_len - 1]

                if len(raw_pcm) > 0:
                    chunk_count += 1
                    total_bytes += len(raw_pcm)

                    if chunk_count == 1:
                        first_chunk_time = time.time() - request_start

                    yield raw_pcm

            except asyncio.CancelledError:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                        temp_files.remove(temp_wav_path) if temp_wav_path in temp_files else None
                    except:
                        pass

                if chunk_count == 0:
                    print(f"✗ Client disconnected before first chunk")
                else:
                    print(f"✗ Client disconnected after {chunk_count} chunk(s), {total_bytes:,} bytes sent")
                break

            finally:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                        temp_files.remove(temp_wav_path) if temp_wav_path in temp_files else None
                    except:
                        pass

            if chunk_count % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if chunk_count > 0 and total_bytes > 0:
            total_time = time.time() - request_start
            audio_duration_sec = total_bytes / (FINAL_SAMPLE_RATE * 2)  # 16kHz, 16-bit = 2 bytes per sample
            print(f"✓ {voice}: TTFT={first_chunk_time:.3f}s Total={total_time:.2f}s Audio={audio_duration_sec:.2f}s Chunks={chunk_count} Bytes={total_bytes:,}")
        else:
            print(f"⚠️  {voice}: No audio generated! Input text may be empty or encoding failed.")

    except Exception as e:
        print(f"✗ Error in streaming generator after {chunk_count} chunks: {e}")
        import traceback
        traceback.print_exc()

    finally:
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
    try:
        if not request.voice:
            request.voice = DEFAULT_VOICE
            print(f"⚠️  No voice specified, using default voice: {request.voice}")

        original_voice = request.voice
        if not validate_voice(request.voice):
            request.voice = DEFAULT_VOICE
            print(f"⚠️  Voice '{original_voice}' not found, falling back to default voice: {request.voice}")

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
    voices_with_info = {
        voice_id: {
            **info,
            'id': voice_id,
            'usage_count': voice_usage_stats.get(voice_id, 0),
            'cached': voice_id in voice_context_cache
        }
        for voice_id, info in AVAILABLE_VOICES.items()
    }

    return {
        'total': len(voices_with_info),
        'default_voice': DEFAULT_VOICE,
        'voices_directory': str(VOICES_DIR),
        'voices': voices_with_info
    }

@app.get('/voices/{voice_id}')
async def get_voice_details(voice_id: str):
    if not validate_voice(voice_id):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Use /voices endpoint to see available voices."
        )
    return get_voice_info(voice_id)

@app.get('/voices/refresh')
async def refresh_voices():
    global AVAILABLE_VOICES, DEFAULT_VOICE, voice_usage_stats, voice_context_cache

    old_count = len(AVAILABLE_VOICES)
    AVAILABLE_VOICES = discover_voices()

    if AVAILABLE_VOICES:
        if DEFAULT_VOICE not in AVAILABLE_VOICES:
            DEFAULT_VOICE = list(AVAILABLE_VOICES.keys())[0]
    else:
        DEFAULT_VOICE = None

    new_stats = {voice: voice_usage_stats.get(voice, 0) for voice in AVAILABLE_VOICES}
    voice_usage_stats = new_stats

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
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ FFmpeg found")
    except FileNotFoundError:
        print("FATAL ERROR: FFmpeg is required but not found in the environment PATH.")
        exit(1)

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

    uvicorn.run(app, host='0.0.0.0', port=5100, timeout_keep_alive=30)
