# Quick Start Guide - MiraTTS Streaming Service

Get started with the MiraTTS streaming TTS service in 5 minutes.

## Prerequisites

```bash
# Check if you have Python 3.8+
python3 --version

# Check if you have CUDA GPU
nvidia-smi

# Check if FFmpeg is installed
ffmpeg -version
```

## Installation

### Step 1: Install MiraTTS

```bash
pip install git+https://github.com/ysharma3501/MiraTTS.git
```

### Step 2: Install Service Dependencies

```bash
pip install fastapi uvicorn scipy psutil requests
```

## Setup

### Step 3: Prepare Reference Audio

Create a voices directory and add reference audio files:

```bash
# Create voices directory
mkdir -p voices

# Add your reference audio files
# These can be WAV, MP3, OGG, etc.
# Example:
cp /path/to/your/voice_sample.wav voices/my_voice.wav
```

**Tips for reference audio:**
- 3-10 seconds of clear speech
- Single speaker
- Minimal background noise
- Good quality recording

### Step 4: Start the Service

```bash
python mira_fastapi_service.py
```

You should see:

```
======================================================
MIRATTS FASTAPI SERVER v1.0.0
======================================================
Device: cuda
Model: YatharthS/MiraTTS
Available voices: 1 reference audio file(s)
Default voice: my_voice
Voice directory: ./voices
Listening on: http://0.0.0.0:5100
======================================================
```

## Test the Service

### Option 1: Use the Test Client

```bash
python test_mira_service.py
```

This will:
- Check service health
- List available voices
- Test non-streaming generation
- Test streaming generation
- Show statistics

### Option 2: Use curl

#### Test non-streaming:

```bash
curl -X POST http://localhost:5100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test.",
    "voice": "my_voice"
  }' \
  --output test.raw

# Convert to WAV and play
ffmpeg -f s16le -ar 16000 -ac 1 -i test.raw test.wav
ffplay test.wav
```

#### Test streaming:

```bash
curl -X POST http://localhost:5100/v1/audio/speech-stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is streaming. Each sentence comes fast. Low latency is great.",
    "voice": "my_voice"
  }' \
  --output stream.raw

# Convert and play
ffmpeg -f s16le -ar 16000 -ac 1 -i stream.raw stream.wav
ffplay stream.wav
```

### Option 3: Use Python Client

```python
import requests
import numpy as np
import scipy.io.wavfile as wav

# Non-streaming request
response = requests.post(
    'http://localhost:5100/v1/audio/speech',
    json={
        'input': 'Hello, this is MiraTTS speaking.',
        'voice': 'my_voice'
    }
)

# Save audio
audio = np.frombuffer(response.content, dtype=np.int16)
wav.write('output.wav', 16000, audio)
print(f"Generated {len(audio)/16000:.2f} seconds of audio")
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/audio/speech` | POST | Generate complete audio (non-streaming) |
| `/v1/audio/speech-stream` | POST | Generate audio chunks (streaming) |
| `/voices` | GET | List available voices |
| `/voices/{voice_id}` | GET | Get voice details |
| `/health` | GET | Service health check |
| `/stats` | GET | Usage statistics |
| `/docs` | GET | Interactive API docs |

## Request Format

```json
{
  "input": "Text to convert to speech",
  "voice": "voice_id",
  "tempo": 1.1,
  "volume": 0.8
}
```

**Parameters:**
- `input` (required): Text to synthesize
- `voice` (required): Voice ID (reference audio filename without extension)
- `tempo` (optional): Playback speed multiplier (default: 1.1)
- `volume` (optional): Volume gain (default: 0.8)

## Response Format

Raw PCM audio:
- Format: s16le (signed 16-bit little-endian)
- Sample rate: 16000 Hz
- Channels: 1 (mono)

## Common Issues

### 1. No voices found

**Problem:**
```
WARNING: No reference audio files found!
```

**Solution:**
```bash
# Add reference audio files to voices directory
cp your_audio.wav voices/
curl http://localhost:5100/voices/refresh
```

### 2. FFmpeg not found

**Problem:**
```
FATAL ERROR: FFmpeg is required but not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 3. CUDA out of memory

**Problem:**
```
torch.cuda.OutOfMemoryError
```

**Solution:**
```bash
# Clear cache
curl -X POST http://localhost:5100/voices/clear-cache

# Or restart the service
```

### 4. Slow first request

**Explanation:** The first request per voice is slower because:
1. Reference audio needs to be encoded (~100-200ms)
2. GPU needs to warm up

**Solution:** This is normal. Subsequent requests will be faster due to context caching.

## Production Deployment

For production use with multiple workers:

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker mira_fastapi_service:app \
  --bind 0.0.0.0:5100 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile -
```

## Performance Tips

1. **Use context caching**: Keep the service running to benefit from cached voice encodings
2. **Pre-warm voices**: Make a test request for each voice after startup
3. **Optimize reference audio**: Use shorter, cleaner samples (3-5 seconds)
4. **Monitor memory**: Check `/health` endpoint regularly
5. **Use streaming**: For long texts, streaming provides better user experience

## Next Steps

- Read the full documentation: `MIRA_SERVICE_README.md`
- Explore the API: http://localhost:5100/docs
- Add more voices to the voices directory
- Integrate with your application

## Support

- **MiraTTS Model**: https://huggingface.co/YatharthS/MiraTTS
- **Service Issues**: Check logs and `/health` endpoint
- **Audio Quality**: Ensure reference audio is high quality

## Example: Real-time Conversation

```python
import requests
import pyaudio
import numpy as np

def stream_tts(text, voice="my_voice"):
    """Stream TTS audio and play in real-time"""
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
        output=True
    )

    # Stream and play
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            stream.write(chunk)

    stream.close()
    p.terminate()

# Use it
stream_tts("Hello, this is real-time streaming text to speech!")
```

That's it! You're ready to use MiraTTS streaming service. 🎉
