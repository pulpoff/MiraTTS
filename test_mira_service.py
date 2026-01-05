#!/usr/bin/env python3
"""
Test client for MiraTTS FastAPI Service

Tests both streaming and non-streaming endpoints.
"""

import requests
import numpy as np
import scipy.io.wavfile as wav
import time
import argparse
from pathlib import Path


BASE_URL = "http://localhost:5100"


def test_list_voices():
    """Test listing available voices"""
    print("\n" + "="*60)
    print("TEST: List Available Voices")
    print("="*60)

    response = requests.get(f"{BASE_URL}/voices")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total']} voice(s)")
        print(f"  Default voice: {data['default_voice']}")
        print(f"  Voices directory: {data['voices_directory']}")
        print("\nAvailable voices:")
        for voice_id, info in data['voices'].items():
            cached_status = "✓ cached" if info['cached'] else "not cached"
            print(f"  • {voice_id}: {info['name']} ({info['format']}, {info['file_size_mb']}MB) [{cached_status}]")
        return data['default_voice']
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return None


def test_non_streaming(voice, text, output_file="test_non_streaming.wav"):
    """Test non-streaming endpoint"""
    print("\n" + "="*60)
    print("TEST: Non-Streaming Generation")
    print("="*60)
    print(f"Voice: {voice}")
    print(f"Text: {text}")

    start_time = time.time()

    response = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            'input': text,
            'voice': voice,
            'tempo': 1.1,
            'volume': 0.8
        }
    )

    end_time = time.time()

    if response.status_code == 200:
        # Convert to numpy array
        audio_data = np.frombuffer(response.content, dtype=np.int16)

        # Save as WAV
        wav.write(output_file, 16000, audio_data)

        duration = len(audio_data) / 16000
        print(f"✓ Success!")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Generation time: {end_time - start_time:.2f}s")
        print(f"  Real-time factor: {duration / (end_time - start_time):.2f}x")
        print(f"  Audio size: {len(response.content):,} bytes")
        print(f"  Saved to: {output_file}")

        # Show headers
        print(f"  Voice used: {response.headers.get('X-Voice-Used')}")
        print(f"  Voice name: {response.headers.get('X-Voice-Name')}")

        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_streaming(voice, text, output_file="test_streaming.wav"):
    """Test streaming endpoint"""
    print("\n" + "="*60)
    print("TEST: Streaming Generation")
    print("="*60)
    print(f"Voice: {voice}")
    print(f"Text: {text}")

    start_time = time.time()
    first_byte_time = None

    response = requests.post(
        f"{BASE_URL}/v1/audio/speech-stream",
        json={
            'input': text,
            'voice': voice,
            'tempo': 1.1,
            'volume': 0.8
        },
        stream=True
    )

    if response.status_code == 200:
        chunks = []
        chunk_count = 0

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                if first_byte_time is None:
                    first_byte_time = time.time()
                    print(f"✓ First byte received in {first_byte_time - start_time:.3f}s")

                chunks.append(chunk)
                chunk_count += 1
                print(f"  Received chunk {chunk_count}: {len(chunk):,} bytes")

        end_time = time.time()

        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(chunks), dtype=np.int16)

        # Save as WAV
        wav.write(output_file, 16000, audio_data)

        duration = len(audio_data) / 16000
        print(f"✓ Streaming complete!")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Total time: {end_time - start_time:.2f}s")
        print(f"  First chunk latency: {first_byte_time - start_time:.3f}s")
        print(f"  Real-time factor: {duration / (end_time - start_time):.2f}x")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Audio size: {len(audio_data) * 2:,} bytes")
        print(f"  Saved to: {output_file}")

        # Show headers
        print(f"  Voice used: {response.headers.get('X-Voice-Used')}")
        print(f"  Voice name: {response.headers.get('X-Voice-Name')}")

        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_voice_details(voice):
    """Test getting voice details"""
    print("\n" + "="*60)
    print(f"TEST: Get Voice Details ({voice})")
    print("="*60)

    response = requests.get(f"{BASE_URL}/voices/{voice}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Voice details retrieved")
        print(f"  ID: {data['id']}")
        print(f"  Name: {data['name']}")
        print(f"  Format: {data['format']}")
        print(f"  File size: {data['file_size_mb']}MB")
        print(f"  Usage count: {data['usage_count']}")
        print(f"  Cached: {data['cached']}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_stats():
    """Test service statistics"""
    print("\n" + "="*60)
    print("TEST: Service Statistics")
    print("="*60)

    response = requests.get(f"{BASE_URL}/stats")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Statistics retrieved")
        print(f"  Total requests: {data['total_requests']}")
        print(f"  Available voices: {data['available_voices']}")
        print(f"  Cached voices: {data['cached_voices']}")
        print(f"  Default voice: {data['default_voice']}")

        if data.get('most_used_voice'):
            most_used = data['most_used_voice']
            print(f"  Most used voice: {most_used['id']} ({most_used['usage_count']} times)")

        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_health():
    """Test health check"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Service is healthy")
        print(f"  Status: {data['status']}")
        print(f"  Requests processed: {data['requests_processed']}")
        print(f"  Memory usage: {data.get('memory_percent', 'N/A')}%")

        if 'gpu_memory_allocated_mb' in data:
            print(f"  GPU memory allocated: {data['gpu_memory_allocated_mb']}MB")
            print(f"  GPU memory cached: {data['gpu_memory_cached_mb']}MB")

        return True
    else:
        print(f"✗ Service unhealthy: {response.status_code}")
        print(response.text)
        return False


def main():
    parser = argparse.ArgumentParser(description='Test MiraTTS FastAPI Service')
    parser.add_argument('--url', default=BASE_URL, help='Service URL')
    parser.add_argument('--voice', help='Voice ID to use (default: first available)')
    parser.add_argument('--text', default=None, help='Custom text to synthesize')
    parser.add_argument('--test', choices=['all', 'health', 'voices', 'non-streaming', 'streaming', 'stats'],
                       default='all', help='Which test to run')

    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip('/')

    print("="*60)
    print("MiraTTS FastAPI Service - Test Client")
    print("="*60)
    print(f"Service URL: {BASE_URL}")

    # Test text samples
    short_text = "Hello, this is a test of the MiraTTS system."
    long_text = (
        "This is a longer text for testing streaming functionality. "
        "It contains multiple sentences. Each sentence will be processed separately. "
        "This allows for lower latency in streaming mode. "
        "The first sentence can be heard while later sentences are still being generated."
    )

    test_text = args.text if args.text else long_text

    try:
        # Test health first
        if args.test in ['all', 'health']:
            test_health()

        # List voices
        if args.test in ['all', 'voices']:
            default_voice = test_list_voices()
        else:
            # Get default voice
            response = requests.get(f"{BASE_URL}/voices")
            default_voice = response.json()['default_voice'] if response.status_code == 200 else None

        if not default_voice:
            print("\n✗ No voices available. Cannot proceed with TTS tests.")
            print("Please add reference audio files to the voices directory.")
            return

        voice = args.voice or default_voice

        # Test voice details
        if args.test in ['all', 'voices']:
            test_voice_details(voice)

        # Test non-streaming
        if args.test in ['all', 'non-streaming']:
            test_non_streaming(voice, test_text, "test_non_streaming.wav")

        # Test streaming
        if args.test in ['all', 'streaming']:
            test_streaming(voice, test_text, "test_streaming.wav")

        # Test stats
        if args.test in ['all', 'stats']:
            test_stats()

        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to service at {BASE_URL}")
        print("Make sure the service is running:")
        print("  python mira_fastapi_service.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
