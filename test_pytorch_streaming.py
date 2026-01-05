#!/usr/bin/env python3
"""Test PyTorch backend token-level streaming implementation"""

import time
import torch
import scipy.io.wavfile as wav
import numpy as np
from mira.streaming_model import MiraTTSStreaming

def test_pytorch_streaming():
    print("🚀 Initializing MiraTTSStreaming with PyTorch backend...")
    start = time.time()
    tts = MiraTTSStreaming()
    init_time = time.time() - start
    print(f"✓ Initialized in {init_time:.2f}s")

    # Encode reference audio
    print("\n🎤 Encoding reference audio...")
    reference_audio = "voices/emily_en.wav"
    context_tokens = tts.encode_audio(reference_audio)
    print(f"✓ Encoded {len(context_tokens)} context tokens")

    # Test text
    test_text = "Hello, this is a test of the new PyTorch token-level streaming implementation. Let's see if it works better than before."
    print(f"\n📝 Test text: {test_text}")
    print(f"   Length: {len(test_text)} characters")

    # Test streaming generation
    print("\n🎵 Generating audio with token-level streaming...")
    audio_chunks = []
    chunk_count = 0
    first_chunk_time = None
    start_time = time.time()

    try:
        for audio_chunk in tts.stream_generate(test_text, context_tokens, chunk_size=50):
            chunk_count += 1

            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
                print(f"✓ TTFT: {first_chunk_time*1000:.0f}ms (chunk {chunk_count})")
            else:
                print(f"  Chunk {chunk_count}: {len(audio_chunk)} samples")

            audio_chunks.append(audio_chunk)

    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return

    total_time = time.time() - start_time

    # Combine audio chunks
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=0)

        # Save to file
        output_file = "test_pytorch_streaming.wav"
        audio_numpy = full_audio.cpu().numpy() if hasattr(full_audio, 'cpu') else full_audio

        # Convert to int16
        if audio_numpy.dtype in [np.float16, np.float32, np.float64]:
            audio_numpy = audio_numpy.astype(np.float32)
            audio_numpy = (audio_numpy * 32767).astype(np.int16)

        wav.write(output_file, 24000, audio_numpy)

        # Calculate metrics
        audio_duration_sec = len(full_audio) / 24000
        rtf = total_time / audio_duration_sec if audio_duration_sec > 0 else 0

        print(f"\n✅ SUCCESS!")
        print(f"   TTFT: {first_chunk_time*1000:.0f}ms")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Audio duration: {audio_duration_sec:.2f}s")
        print(f"   RTF: {rtf:.2f}x")
        print(f"   Chunks: {chunk_count}")
        print(f"   Output: {output_file}")
    else:
        print("❌ No audio chunks generated")

if __name__ == "__main__":
    test_pytorch_streaming()
