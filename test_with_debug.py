#!/usr/bin/env python3
"""
Test client that sends a long sentence and captures the server response with full debug info.
"""
import urllib.request
import json
import sys
import time

url = 'http://localhost:8000/animate'
text = ("The quick brown fox jumps over the lazy dog while the mysterious wind whispers "
        "through the ancient forest, carrying tales of forgotten civilizations and lost treasures "
        "hidden beneath the moonlit sky.")

print(f"Sending request to {url}")
print(f"Text: {text[:80]}...")
print(f"Text length: {len(text)} chars, {len(text.split())} words\n")

data = json.dumps({'text': text}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    print("Waiting for response...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        print(f"✓ Response received: status {resp.status}, size {len(body)} bytes\n")
        
        try:
            j = json.loads(body)
            print("RESPONSE SUMMARY:")
            print(f"  - frames: {len(j.get('frames', []))}")
            print(f"  - fps: {j.get('fps', '?')}")
            print(f"  - audio_url: {j.get('audio_url', '?')}")
            print(f"  - num_phonemes: {j.get('num_phonemes', '?')}")
            
            # Compute expected frames
            import os
            audio_file = j.get('audio_url', '').lstrip('/')
            if os.path.exists(audio_file):
                from pydub import AudioSegment
                try:
                    audio = AudioSegment.from_mp3(audio_file)
                    duration_sec = len(audio) / 1000.0
                    fps = j.get('fps', 30)
                    expected_frames = int(duration_sec * fps)
                    actual_frames = len(j.get('frames', []))
                    print(f"\nAUDIO ANALYSIS:")
                    print(f"  - audio file: {audio_file}")
                    print(f"  - duration: {duration_sec:.3f}s")
                    print(f"  - expected frames @ {fps}fps: {expected_frames}")
                    print(f"  - actual frames: {actual_frames}")
                    print(f"  - match: {'✓ YES' if actual_frames == expected_frames else '✗ NO'}")
                    if actual_frames != expected_frames:
                        print(f"  - difference: {actual_frames - expected_frames} frames")
                except Exception as e:
                    print(f"  Error analyzing audio: {e}")
        except Exception as e:
            print(f"✗ Parse error: {e}")
            print(body[:500])
except Exception as e:
    print(f"✗ Request failed: {e}")
    sys.exit(1)
