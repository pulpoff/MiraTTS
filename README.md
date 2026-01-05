# MiraTTS v2.0 - Real Chunked Streaming

[MiraTTS](https://huggingface.co/YatharthS/MiraTTS) is a finetune of the excellent [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) model for enhanced realism and stability performing on par with closed source models.

This repository heavily optimizes MiraTTS with [LMDeploy](https://github.com/InternLM/lmdeploy) and boosts quality by using [FlashSR](https://github.com/ysharma3501/FlashSR) to generate high quality audio at over **100x** realtime!

**🆕 v2.0 Features:**
- ✨ **Real chunked streaming** with token-level granularity (100-200ms first chunk latency)
- 🚀 **FastAPI service** with Kokoro TTS-compatible API
- 🎙️ **Voice cloning** via reference audio files
- ⚡ **Low latency** streaming similar to commercial TTS services

https://github.com/user-attachments/assets/262088ae-068a-49f2-8ad6-ab32c66dcd17

## Key Benefits

- **Incredibly fast**: Over 100x realtime using LMDeploy and batching
- **High quality**: Generates clear and crisp 48kHz audio outputs
- **Memory efficient**: Works within 6GB VRAM
- **Low latency**: First chunk in ~100-200ms with streaming
- **Real chunked streaming**: Token-level streaming for smooth audio delivery
- **Voice cloning**: Clone any voice from a reference audio sample
- **Production ready**: FastAPI service with full API compatibility

## Installation

### Quick Install (Library Only - Original MiraTTS)

```bash
pip install git+https://github.com/ysharma3501/MiraTTS.git
```

### Full Install (with FastAPI Service and Streaming)

```bash
# Clone this repository (includes streaming and FastAPI service)
git clone https://github.com/pulpoff/MiraTTS.git
cd MiraTTS

# Install MiraTTS package with all core dependencies (ncodec, fastaudiosr, etc.)
pip install -e .

# Install additional service dependencies
pip install -r requirements.txt
```

**Note**: The `-e` flag installs in editable mode, which is recommended for development. For production, you can omit it: `pip install .`

### System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, 6GB+ VRAM)
- FFmpeg (required for audio processing)

**Install FFmpeg:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Option 1: Direct Python Usage (Source)

#### Basic Generation
```python
from mira.model import MiraTTS
from IPython.display import Audio

# Initialize MiraTTS
mira_tts = MiraTTS('YatharthS/MiraTTS')

# Reference audio file (clone this voice)
reference_file = "voices/john.wav"  # Can be mp3/wav/ogg or anything librosa supports

# Reference text (transcript of reference audio) - IMPORTANT for voice cloning!
# This is what was actually said in john.wav
reference_text = "Hello, my name is John and I'm demonstrating voice cloning."

# Text to synthesize
text = "Alright, so have you ever heard of a little thing named text to speech? Well, it allows you to convert text into speech! I know, that's super cool, isn't it?"

# Encode reference audio
context_tokens = mira_tts.encode_audio(reference_file)

# Generate speech with reference text for better cloning
audio = mira_tts.generate(text, context_tokens, reference_text=reference_text)

# Play or save
Audio(audio, rate=48000)
```

#### Streaming Generation (New in v2.0!)
```python
from mira.streaming_model import MiraTTSStreaming
import scipy.io.wavfile as wav

# Initialize streaming model
mira_tts = MiraTTSStreaming('YatharthS/MiraTTS')

# Reference audio and text
reference_file = "voices/daniel.wav"
reference_text = "Hi, I'm Daniel. This is my voice sample for cloning."
context_tokens = mira_tts.encode_audio(reference_file)

# Text to synthesize
text = "This is streaming generation. You'll get audio chunks as they're generated, providing low latency!"

# Generate with streaming (yields chunks as tokens are produced)
chunks = []
for audio_chunk in mira_tts.stream_generate(text, context_tokens, chunk_size=50, reference_text=reference_text):
    chunks.append(audio_chunk.cpu().numpy())
    # Process each chunk immediately (e.g., stream to client, play audio, etc.)
    print(f"Received chunk: {len(audio_chunk)} samples")

# Combine all chunks
import numpy as np
full_audio = np.concatenate(chunks)

# Save
wav.write('output.wav', 48000, full_audio)
```

#### Batch Generation
```python
# Multiple texts with same voice
texts = [
    "Hey, what's up! I am feeling SO happy!",
    "Honestly, this is really interesting, isn't it?"
]

reference_file = "voices/john.wav"
context_tokens = [mira_tts.encode_audio(reference_file)]

# Generate all at once
audio = mira_tts.batch_generate(texts, context_tokens)

Audio(audio, rate=48000)
```

### Option 2: FastAPI Service (Production)

#### Starting the Service

```bash
# Place reference audio files in the voices directory
mkdir -p voices
cp your_voice_samples/*.wav voices/

# Start the service
python mira_fastapi_service.py
```

The service will start on `http://0.0.0.0:5100`

#### API Endpoints

**Available endpoints:**
- `POST /v1/audio/speech` - Generate complete audio (non-streaming)
- `POST /v1/audio/speech-stream` - Stream audio chunks (low latency)
- `GET /voices` - List available reference voices
- `GET /health` - Service health check
- `GET /docs` - Interactive API documentation

#### API Usage Examples

##### Non-Streaming Generation

```bash
# Using curl
curl -X POST http://localhost:5100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is MiraTTS speaking with real-time voice cloning!",
    "voice": "john",
    "tempo": 1.1,
    "volume": 0.8
  }' \
  --output speech.raw

# Convert to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i speech.raw speech.wav
```

##### Streaming Generation (Low Latency)

```bash
# Using curl - Audio starts streaming in ~100-200ms!
curl -X POST http://localhost:5100/v1/audio/speech-stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is streaming mode. The audio starts playing almost immediately!",
    "voice": "daniel",
    "tempo": 1.0,
    "volume": 0.8
  }' \
  --output stream.raw

# Convert to WAV
ffmpeg -f s16le -ar 16000 -ac 1 -i stream.raw stream.wav
```

##### Python Client Example

```python
import requests
import numpy as np
import scipy.io.wavfile as wav

# Non-streaming request
response = requests.post(
    'http://localhost:5100/v1/audio/speech',
    json={
        'input': 'Hello from MiraTTS!',
        'voice': 'john',  # Reference file: ref/john.wav
        'tempo': 1.1,
        'volume': 0.8
    }
)

# Save audio
audio = np.frombuffer(response.content, dtype=np.int16)
wav.write('output.wav', 16000, audio)
print(f"Generated {len(audio)/16000:.2f} seconds of audio")
```

##### Real-time Streaming with Playback

```python
import requests
import pyaudio
import time

def stream_and_play(text, voice="john"):
    """Stream TTS and play audio in real-time"""

    start_time = time.time()

    response = requests.post(
        'http://localhost:5100/v1/audio/speech-stream',
        json={'input': text, 'voice': voice},
        stream=True
    )

    # Setup audio player
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        output=True,
        frames_per_buffer=4096
    )

    first_chunk = True

    # Stream and play chunks as they arrive
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            if first_chunk:
                latency = (time.time() - start_time) * 1000
                print(f"✓ First chunk in {latency:.0f}ms - Audio playing!")
                first_chunk = False
            stream.write(chunk)

    stream.close()
    p.terminate()
    print("✓ Streaming complete")

# Usage - Audio starts playing in ~100-200ms!
stream_and_play("This is real-time streaming with minimal latency!")
```

##### List Available Voices

```bash
# List all reference voices
curl http://localhost:5100/voices

# Example response:
# {
#   "total": 2,
#   "default_voice": "john",
#   "voices": {
#     "john": {
#       "name": "John",
#       "path": "/ref/john.wav",
#       "format": "WAV",
#       "file_size_mb": 0.52,
#       "cached": true
#     },
#     "daniel": {
#       "name": "Daniel",
#       "path": "/ref/daniel.wav",
#       "format": "WAV",
#       "file_size_mb": 0.48,
#       "cached": true
#     }
#   }
# }
```

##### Health Check

```bash
curl http://localhost:5100/health

# Example response:
# {
#   "status": "healthy",
#   "service": "MiraTTS FastAPI Server v2.0.0 (Real Chunked Streaming)",
#   "available_voices": 2,
#   "cached_voices": 2,
#   "gpu_memory_allocated_mb": 5234.5
# }
```

### Reference Audio Files

Place your reference audio files in the `voices/` directory along with their text transcripts:

```
voices/
├── john.wav      # Male voice sample
├── john.txt      # Transcript of john.wav (IMPORTANT for voice cloning!)
├── daniel.wav    # Another male voice
├── daniel.txt    # Transcript of daniel.wav
├── sarah.wav     # Female voice sample
├── sarah.txt     # Transcript of sarah.wav
└── emma.wav      # Another female voice
└── emma.txt      # Transcript of emma.wav
```

**Reference Text Files (IMPORTANT!):**
- Each `.wav` file should have a corresponding `.txt` file with the same name
- The `.txt` file contains the exact transcript of what's said in the audio
- This significantly improves voice cloning quality
- Example: If `john.wav` contains "Hello, my name is John", then `john.txt` should contain: `Hello, my name is John`

**Tips for best results:**
- Use 3-10 seconds of clean, clear speech
- Single speaker recordings work best
- Avoid background noise
- **Always provide reference text** for better cloning quality
- Supported formats: WAV, MP3, OGG, FLAC, M4A

The voice ID is the filename without extension (e.g., `voices/john.wav` + `voices/john.txt` → voice ID: `john`)

## API Request Format

```json
{
  "input": "Text to convert to speech",
  "voice": "john",
  "tempo": 1.1,
  "volume": 0.8
}
```

**Parameters:**
- `input` (required): Text to synthesize
- `voice` (required): Voice ID (reference audio filename without extension)
- `tempo` (optional): Playback speed multiplier (default: 1.1)
- `volume` (optional): Volume gain (default: 0.8)

**Response format:**
- Raw PCM audio: s16le (signed 16-bit little-endian)
- Sample rate: 16000 Hz
- Channels: 1 (mono)

## Performance

- **100x+ realtime** generation speed
- First chunk in **~100-200ms** (streaming mode)
- Full generation: ~1-2 seconds for 10 seconds of audio

## Production Deployment

For production use with multiple workers:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mira_fastapi_service:app \
  --bind 0.0.0.0:5100 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile -
```

## Documentation

- **QUICKSTART.md** - Get started in 5 minutes
- **MIRA_SERVICE_README.md** - Complete API reference and service documentation
- **REAL_CHUNKED_STREAMING.md** - Deep dive into streaming implementation
- **KOKORO_VS_MIRA.md** - Comparison with Kokoro TTS

## Examples

See the [HuggingFace model page](https://huggingface.co/YatharthS/MiraTTS) for audio samples and demos.

## Learning Resources

I recommend reading these 2 blogs to better understand LLM TTS models and optimization:
- How they work: https://huggingface.co/blog/YatharthS/llm-tts-models
- How to optimize them: https://huggingface.co/blog/YatharthS/making-neutts-200x-realtime

## Training

Released training code! You can now train the model to be multilingual, multi-speaker, or support audio events on any local or cloud GPU!

- **Kaggle notebook**: https://www.kaggle.com/code/yatharthsharma888/miratts-training
- **Colab notebook**: https://colab.research.google.com/drive/1IprDyaMKaZrIvykMfNrxWFeuvj-DQPII?usp=sharing

## Roadmap

- [x] Release code and model
- [x] Release training code
- [x] **Support low latency streaming** ✨ **NEW in v2.0!**
- [x] **FastAPI service with real chunked streaming** ✨ **NEW in v2.0!**
- [ ] Release native 48kHz bicodec
- [ ] GPU-accelerated codec decoding
- [ ] Multi-request batching

## What's New in v2.0

### Real Chunked Streaming
- Token-level streaming using LMDeploy's `stream_infer()`
- **100-200ms first chunk latency** (5-10x faster than sentence-based)
- Configurable chunk size for latency vs efficiency tuning
- Similar streaming behavior to Kokoro and MeloTTS

### FastAPI Production Service
- Full REST API with `/v1/audio/speech` and `/v1/audio/speech-stream` endpoints
- Kokoro TTS API compatibility
- Voice context caching for performance
- Production-ready with Gunicorn support

### Voice Cloning
- Use any audio file as a reference voice
- Automatic voice discovery from `ref/` directory
- Support for WAV, MP3, OGG, FLAC, M4A formats
- Unlimited custom voices

## Architecture

```
User Request
     ↓
FastAPI Service
     ↓
MiraTTSStreaming.stream_generate()
     ↓
LMDeploy.stream_infer() → [tokens streaming...]
     ↓
Incremental Audio Decoding (every N tokens)
     ↓
FFmpeg Processing (tempo, volume, resampling)
     ↓
Stream to Client → Audio plays in ~100-200ms!
```

## Testing

Test the service with the included test client:

```bash
# Run all tests
python test_mira_service.py

# Test specific features
python test_mira_service.py --test streaming
python test_mira_service.py --test health
python test_mira_service.py --voice john
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Credits

- **MiraTTS**: Enhanced model based on Spark-TTS
- **Spark-TTS**: Base model architecture
- **LMDeploy**: Fast inference engine
- **FlashSR**: Audio super-resolution
- **unsloth**: Training optimizations

Thanks very much to the authors of Spark-TTS and unsloth. Thanks for checking out this repository as well.

Stars would be well appreciated, thank you! ⭐

## Contact

Email: yatharthsharma3501@gmail.com

## License

Please refer to the model page on HuggingFace for license information.
