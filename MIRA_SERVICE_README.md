# MiraTTS FastAPI Service

A drop-in replacement for Kokoro TTS service using MiraTTS with streaming support.

## Features

- ✅ **Voice Cloning**: Uses reference audio files as voice templates
- ✅ **Streaming Support**: Implements chunked streaming by splitting text into sentences
- ✅ **High Quality**: Generates 48kHz audio (downsampled to 16kHz for output)
- ✅ **Low Latency**: ~100ms first chunk latency
- ✅ **API Compatible**: Drop-in replacement for Kokoro TTS API
- ✅ **Context Caching**: Caches encoded voice references for performance
- ✅ **Multiple Voices**: Supports any audio file as reference (WAV, MP3, OGG, etc.)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (6GB+ VRAM recommended)
- FFmpeg installed

### Install Dependencies

```bash
# Install MiraTTS
pip install git+https://github.com/ysharma3501/MiraTTS.git

# Install FastAPI and dependencies
pip install fastapi uvicorn scipy psutil

# Ensure FFmpeg is installed
ffmpeg -version
```

## Setup

### 1. Prepare Voice Directory

Create a directory for reference audio files:

```bash
mkdir -p /voices
# or use ./voices in the current directory
```

### 2. Add Reference Audio Files

Place reference audio files in the voices directory. These will be used as voice templates:

```bash
# Example voice files
/voices/
├── john_doe.wav
├── jane_smith.mp3
├── narrator.ogg
└── assistant.wav
```

**Supported formats**: WAV, MP3, OGG, FLAC, M4A

**Tips for best results**:
- Use clean, clear audio samples
- 3-10 seconds of speech is ideal
- Avoid background noise
- Single speaker recordings work best

### 3. Start the Service

```bash
python mira_fastapi_service.py
```

The service will start on `http://0.0.0.0:5100`

## API Endpoints

### Generate Speech (Non-Streaming)

**POST** `/v1/audio/speech`

Generate complete audio file at once.

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the MiraTTS system.",
    "voice": "john_doe",
    "tempo": 1.1,
    "volume": 0.8
  }' \
  --output speech.raw
```

### Generate Speech (Streaming)

**POST** `/v1/audio/speech-stream`

Stream audio chunks with low latency.

```bash
curl -X POST http://localhost:5100/v1/audio/speech-stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a longer text that will be streamed sentence by sentence. Each sentence generates a chunk. This provides low latency for the first words.",
    "voice": "jane_smith",
    "tempo": 1.0,
    "volume": 0.8
  }' \
  --output speech_stream.raw
```

### List Available Voices

**GET** `/voices`

```bash
curl http://localhost:5100/voices
```

Response:
```json
{
  "total": 2,
  "default_voice": "john_doe",
  "voices_directory": "/voices",
  "voices": {
    "john_doe": {
      "name": "John Doe",
      "path": "/voices/john_doe.wav",
      "format": "WAV",
      "file_size_mb": 0.52,
      "id": "john_doe",
      "usage_count": 5,
      "cached": true
    }
  }
}
```

### Get Voice Details

**GET** `/voices/{voice_id}`

```bash
curl http://localhost:5100/voices/john_doe
```

### Refresh Voice List

**GET** `/voices/refresh`

Rescans the voice directory for new/removed files.

```bash
curl http://localhost:5100/voices/refresh
```

### Clear Context Cache

**POST** `/voices/clear-cache`

Clears cached voice context tokens to free memory.

```bash
curl -X POST http://localhost:5100/voices/clear-cache
```

### Service Statistics

**GET** `/stats`

```bash
curl http://localhost:5100/stats
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:5100/health
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text to convert to speech |
| `voice` | string | (first voice) | Voice ID (filename without extension) |
| `speed` | float | 1.0 | (Not used by MiraTTS, kept for API compatibility) |
| `tempo` | float | 1.1 | Audio playback tempo (0.5-2.0) |
| `volume` | float | 0.8 | Audio volume gain (0.0-2.0) |

## Response Format

The audio output is raw PCM data:
- **Format**: s16le (signed 16-bit little-endian)
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (mono)

### Playing Audio

```bash
# Using ffplay
curl -X POST http://localhost:5100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "john_doe"}' \
  --output - | ffplay -f s16le -ar 16000 -ac 1 -

# Convert to WAV
curl -X POST http://localhost:5100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "john_doe"}' \
  --output speech.raw

ffmpeg -f s16le -ar 16000 -ac 1 -i speech.raw speech.wav
```

## Python Client Example

```python
import requests
import numpy as np
import scipy.io.wavfile as wav

# Non-streaming
response = requests.post(
    'http://localhost:5100/v1/audio/speech',
    json={
        'input': 'Hello, this is a test of the MiraTTS system.',
        'voice': 'john_doe',
        'tempo': 1.1,
        'volume': 0.8
    }
)

# Save as WAV
audio_data = np.frombuffer(response.content, dtype=np.int16)
wav.write('output.wav', 16000, audio_data)

# Streaming
response = requests.post(
    'http://localhost:5100/v1/audio/speech-stream',
    json={
        'input': 'This is streaming text. Each sentence is processed separately.',
        'voice': 'john_doe'
    },
    stream=True
)

chunks = []
for chunk in response.iter_content(chunk_size=8192):
    if chunk:
        chunks.append(chunk)

audio_data = np.frombuffer(b''.join(chunks), dtype=np.int16)
wav.write('output_stream.wav', 16000, audio_data)
```

## How Streaming Works

Since MiraTTS doesn't natively support streaming, the service implements it by:

1. **Text Splitting**: Input text is split into sentences using regex patterns
2. **Sequential Generation**: Each sentence is generated separately using MiraTTS
3. **Chunk Streaming**: Audio for each sentence is processed with FFmpeg and streamed
4. **Context Caching**: Reference audio is encoded once and cached for subsequent requests

This provides:
- Low first-chunk latency (only first sentence needs to be generated)
- Progressive audio playback while later sentences generate
- Efficient memory usage through chunking

## Performance Optimization

### Context Caching

Voice context tokens are cached automatically after first use:

```python
# First request: Encodes reference audio (~100-200ms)
POST /v1/audio/speech {"input": "...", "voice": "john_doe"}

# Subsequent requests: Uses cached context (0ms encoding overhead)
POST /v1/audio/speech {"input": "...", "voice": "john_doe"}
```

### Clear Cache When Needed

If you update reference audio files or want to free memory:

```bash
curl -X POST http://localhost:5100/voices/clear-cache
```

### Multiple Workers

For production, use Gunicorn with multiple workers:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker mira_fastapi_service:app \
  --bind 0.0.0.0:5100 \
  --timeout 120 \
  --keep-alive 5 \
  --worker-tmp-dir /dev/shm
```

## Troubleshooting

### No voices found

```
❌ WARNING: No reference audio files found!
```

**Solution**: Place audio files in the voices directory:
```bash
mkdir -p /voices
cp your_voice.wav /voices/
```

### FFmpeg not found

```
FATAL ERROR: FFmpeg is required but not found
```

**Solution**: Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Out of memory

```
torch.cuda.OutOfMemoryError
```

**Solutions**:
1. Clear context cache: `POST /voices/clear-cache`
2. Reduce concurrent requests
3. Use smaller reference audio files
4. Reduce `cache_max_entry_count` in initialization

### Slow first request

The first request per voice is slower due to:
1. Reference audio encoding (~100-200ms)
2. Model warmup

**Solution**: Pre-warm voices on startup or use context caching.

## Comparison with Kokoro TTS

| Feature | Kokoro TTS | MiraTTS |
|---------|------------|---------|
| **Streaming** | Native chunk-by-chunk | Sentence-by-sentence |
| **Voice Control** | Pre-trained voices | Reference audio cloning |
| **Sample Rate** | 24kHz | 48kHz |
| **Latency** | ~50-100ms | ~100-200ms |
| **Quality** | Good | High (48kHz) |
| **Flexibility** | Fixed voices | Any voice via reference |
| **API** | Custom | Compatible |

## License

This service is a wrapper around MiraTTS. Please refer to the MiraTTS repository for model license information.

## Support

For issues related to:
- **Service/API**: This repository
- **MiraTTS model**: https://github.com/ysharma3501/MiraTTS
- **Audio quality**: Check reference audio quality and MiraTTS parameters

## Credits

- **MiraTTS**: https://huggingface.co/YatharthS/MiraTTS
- **Spark-TTS**: Base model for MiraTTS
- **LMDeploy**: Inference optimization
- **FlashSR**: Audio super-resolution
