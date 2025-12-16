import http.server
import socketserver
import json
import torch
import numpy as np
import os
import tempfile
from gtts import gTTS
from pydub import AudioSegment
from g2p_en import G2p
from model_phoneme import create_phoneme_model

PORT = 8000

# Ensure temp directory exists for audio
AUDIO_DIR = 'temp_audio'
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Load model and vocabulary at startup
print("Initializing phoneme server...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_file = 'phonemes.json'
model_path = 'checkpoints_phoneme/best_model.pt'

# Load phoneme vocabulary
print(f"Loading vocabulary from {vocab_file}...")
with open(vocab_file, 'r') as f:
    vocab_data = json.load(f)
    phoneme_to_id = vocab_data['phoneme_to_id']
    id_to_phoneme = {int(v): k for k, v in phoneme_to_id.items()}
    vocab_size = len(phoneme_to_id)

print(f"Loaded {vocab_size} phonemes")

# Load model
print(f"Loading model from {model_path}...")
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please train the model first using: python train_phoneme.py")
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
model = create_phoneme_model(
    vocab_size=vocab_size,
    embedding_dim=checkpoint['config']['embedding_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# Initialize G2P converter
g2p = G2p()

# Blend shape names (must match the model output order)
BLEND_SHAPE_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
    'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward',
    'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'
]

def text_to_phonemes(text):
    """Convert text to phoneme sequence using g2p_en."""
    phonemes = g2p(text)
    return phonemes

def predict_blend_shapes_phoneme(prev_phoneme_id, curr_phoneme_id, next_phoneme_id, 
                                   frame_pos, duration):
    """Predict blend shapes for a phoneme context."""
    with torch.no_grad():
        # Create batch with single sample
        prev_ph = torch.tensor([prev_phoneme_id], dtype=torch.long, device=device)
        curr_ph = torch.tensor([curr_phoneme_id], dtype=torch.long, device=device)
        next_ph = torch.tensor([next_phoneme_id], dtype=torch.long, device=device)
        frame_p = torch.tensor([frame_pos], dtype=torch.float32, device=device)
        dur = torch.tensor([duration], dtype=torch.float32, device=device)
        
        input_dict = {
            'prev_phoneme': prev_ph,
            'curr_phoneme': curr_ph,
            'next_phoneme': next_ph,
            'frame_pos': frame_p,
            'phoneme_duration': dur
        }
        
        output = model(input_dict)
        blend_shapes = output[0].cpu().numpy().astype(np.float32)
        
    return blend_shapes

def smooth_data(data, window_size=5):
    """
    Apply moving average smoothing to the blend shape data.
    data: (N, 52) numpy array
    window_size: size of the smoothing window (odd number)
    """
    if window_size < 2 or len(data) < window_size:
        return data
        
    smoothed = np.zeros_like(data)
    kernel = np.ones(window_size) / window_size
    pad_size = window_size // 2
    
    for i in range(data.shape[1]):
        channel = data[:, i]
        # Pad with edge values to prevent drops at start/end
        padded = np.pad(channel, (pad_size, pad_size), mode='edge')
        # Convolve
        convolved = np.convolve(padded, kernel, mode='valid')
        # Ensure shape matches
        smoothed[:, i] = convolved[:len(channel)]
        
    return smoothed


def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=5):
    """Return ms of leading silence in an AudioSegment."""
    trim_ms = 0  # ms

    assert chunk_size > 0
    while trim_ms < len(sound):
        chunk = sound[trim_ms:trim_ms+chunk_size]
        # chunk.dBFS can be -inf for absolute silence
        if chunk.dBFS > silence_threshold:
            break
        trim_ms += chunk_size

    return trim_ms


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/animate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            text = data.get('text', '').lower().strip()
            print(f"\n{'='*80}")
            print(f"Received request for: '{text}'")
            print(f"{'='*80}")
            
            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Empty text'}).encode('utf-8'))
                return
            
            # Convert text to phonemes (for logging/debugging)
            print(f"Converting text to phonemes...")
            phonemes = text_to_phonemes(text)
            print(f"Full phonemes (with spaces): {len(phonemes)} total")
            
            # Filter out whitespace tokens immediately (so all downstream uses are consistent)
            phonemes = [p for p in phonemes if str(p).strip() != '']
            print(f"Phonemes (filtered): {phonemes}")
            print(f"Phoneme sequence length (full text): {len(phonemes)}")
            
            # Generate audio in chunks (to allow better control of phoneme durations)
            print(f"Generating audio from TTS in chunks...")
            CHUNK_SIZE = 5  # words per chunk
            words = text.split()
            chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

            combined_audio = AudioSegment.empty()
            phoneme_ids = []
            phoneme_durations = []
            phoneme_durations = []

            for chunk_text in chunks:
                try:
                    tts = gTTS(text=chunk_text, lang='en', slow=False)
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tf:
                        tmp_path = tf.name
                    tts.save(tmp_path)
                    chunk_audio = AudioSegment.from_mp3(tmp_path)
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"Error generating chunk audio for '{chunk_text}': {e}")
                    # fallback to 0.5s silence for the chunk
                    chunk_audio = AudioSegment.silent(duration=500)

                # Trim leading/trailing silence to avoid chunk startup delay
                try:
                    start_trim = detect_leading_silence(chunk_audio, silence_threshold=-40.0, chunk_size=10)
                    end_trim = detect_leading_silence(chunk_audio.reverse(), silence_threshold=-40.0, chunk_size=10)
                    trimmed = chunk_audio[start_trim:len(chunk_audio)-end_trim]
                    if len(trimmed) == 0:
                        # if trimming removed everything, fall back to original
                        trimmed = chunk_audio
                except Exception:
                    trimmed = chunk_audio

                trimmed_duration_sec = len(trimmed) / 1000.0
                print(f"Chunk '{chunk_text}' trimmed: {start_trim}ms lead, {end_trim}ms tail => {trimmed_duration_sec:.3f}s")

                # Append with a small crossfade to avoid gaps
                if len(combined_audio) == 0:
                    combined_audio = trimmed
                else:
                    try:
                        combined_audio = combined_audio.append(trimmed, crossfade=20)
                    except Exception:
                        combined_audio += trimmed

                # Convert chunk text to phonemes and assign equal duration per phoneme
                chunk_phonemes = text_to_phonemes(chunk_text)
                # Filter out whitespace tokens
                filtered = [p for p in chunk_phonemes if str(p).strip() != '']
                num_ph = len(filtered)
                per_ph_dur = trimmed_duration_sec / num_ph if num_ph > 0 else trimmed_duration_sec

                for ph in filtered:
                    ph_id = phoneme_to_id.get(ph, phoneme_to_id.get('<UNK>', 0))
                    phoneme_ids.append(ph_id)
                    phoneme_durations.append(per_ph_dur)
                print(f"  -> {num_ph} phonemes, per_ph_dur={per_ph_dur:.3f}s")

            # Save combined audio
            audio_filename = f"audio_{int(torch.randint(0,1_000_000,(1,)).item())}.mp3"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            try:
                combined_audio.export(audio_path, format="mp3")
            except Exception as e:
                print(f"Error exporting combined audio: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': f'Audio export failed: {e}'}).encode('utf-8'))
                return

            # Measure the ACTUAL exported audio duration by re-loading it
            try:
                exported_audio = AudioSegment.from_mp3(audio_path)
                actual_audio_duration_sec = len(exported_audio) / 1000.0
            except Exception:
                # fallback to combined_audio duration if re-load fails
                actual_audio_duration_sec = len(combined_audio) / 1000.0

            print(f"Actual exported audio duration: {actual_audio_duration_sec:.2f}s")

            # Instead of using per-phoneme durations from chunks (which don't sum to total),
            # distribute the total audio duration equally across all phonemes
            # This ensures frames generated = audio length exactly (no padding needed)
            num_phonemes = len(phoneme_ids)
            if num_phonemes > 0:
                # Equal duration per phoneme across entire audio
                duration_per_phoneme = actual_audio_duration_sec / num_phonemes
                # Recalculate phoneme_durations to be uniform
                phoneme_durations = [duration_per_phoneme] * num_phonemes
                print(f"Redistributing durations: {num_phonemes} phonemes × {duration_per_phoneme:.4f}s = {actual_audio_duration_sec:.2f}s total")

            # Generate animation frames
            all_blend_shapes = []
            fps = 30
            expected_frames = max(1, int(actual_audio_duration_sec * fps))

            print(f"Generating blend shapes using redistributed phoneme durations...")
            print(f"Target frame count: {expected_frames} frames for {actual_audio_duration_sec:.2f}s @ {fps}fps")

            # Distribute frames proportionally across phonemes
            frames_per_phoneme_list = []
            total_assigned = 0
            for i in range(len(phoneme_ids) - 1):
                ph_dur = phoneme_durations[i] if i < len(phoneme_durations) else 0.1
                frames = int(round(ph_dur * fps))
                frames_per_phoneme_list.append(max(1, frames))
                total_assigned += frames_per_phoneme_list[-1]
            
            # Last phoneme gets whatever frames are needed to reach expected_frames
            remaining_frames = expected_frames - total_assigned
            frames_per_phoneme_list.append(max(1, remaining_frames))

            for i, curr_ph_id in enumerate(phoneme_ids):
                prev_ph_id = phoneme_ids[i-1] if i > 0 else phoneme_to_id.get('<START>', 0)
                next_ph_id = phoneme_ids[i+1] if i < len(phoneme_ids)-1 else phoneme_to_id.get('<END>', 0)

                ph_dur = phoneme_durations[i] if i < len(phoneme_durations) else 0.1
                frames_per_phoneme = frames_per_phoneme_list[i] if i < len(frames_per_phoneme_list) else 1

                for f in range(frames_per_phoneme):
                    frame_pos = f / frames_per_phoneme if frames_per_phoneme > 0 else 0.5
                    blend_shapes = predict_blend_shapes_phoneme(
                        prev_ph_id, curr_ph_id, next_ph_id,
                        frame_pos, ph_dur
                    )
                    all_blend_shapes.append(blend_shapes)

            # Ensure frame count matches ACTUAL audio length (pad or trim)
            # Use actual exported audio duration (not trimmed duration)
            expected_frames = max(1, int(actual_audio_duration_sec * fps))
            actual_frames = len(all_blend_shapes)
            print(f"Frames generated: {actual_frames}, expected (actual audio length @ {fps}fps): {expected_frames}")
            
            if actual_frames < expected_frames:
                # pad by repeating last frame
                if actual_frames > 0:
                    last = all_blend_shapes[-1]
                    pad_count = expected_frames - actual_frames
                    for _ in range(pad_count):
                        all_blend_shapes.append(last)
                    print(f"Padded {pad_count} frames to match audio length")
                else:
                    # no frames generated, create neutral frames
                    neutral = np.zeros(52, dtype=np.float32)
                    all_blend_shapes = [neutral for _ in range(expected_frames)]
                    print(f"Generated {expected_frames} neutral frames (no phonemes)")
            elif actual_frames > expected_frames:
                # trim excess
                trim_count = actual_frames - expected_frames
                all_blend_shapes = all_blend_shapes[:expected_frames]
                print(f"Trimmed {trim_count} excess frames")
            
            # Convert to numpy and smooth
            all_blend_shapes = np.array(all_blend_shapes)
            print(f"Generated {len(all_blend_shapes)} frames")
            
            if len(all_blend_shapes) > 0:
                # Apply smoothing
                all_blend_shapes = smooth_data(all_blend_shapes, window_size=5)
                print(f"Applied smoothing filter")
            
            # Convert to dictionary format
            frames = []
            for i in range(len(all_blend_shapes)):
                frame_data = {name: float(val) for name, val in zip(BLEND_SHAPE_NAMES, all_blend_shapes[i])}
                frames.append(frame_data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response_data = {
                'frames': frames, 
                'fps': fps,
                'audio_url': f"/{AUDIO_DIR}/{audio_filename}",
                'phonemes': phonemes,
                'num_phonemes': len(phoneme_ids)
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            print(f"✓ Response sent: {len(frames)} frames")
            
        else:
            # Serve static files
            super().do_GET()

    def do_GET(self):
        # Allow serving files from the temp audio directory
        if self.path.startswith(f"/{AUDIO_DIR}/"):
            if ".." in self.path:
                self.send_error(403)
                return
            super().do_GET()
        else:
            super().do_GET()

print(f"\n{'='*80}")
print(f"✓ PHONEME SERVER READY")
print(f"{'='*80}")
print(f"Serving at http://localhost:{PORT}")
print(f"Visit http://localhost:{PORT}/visualizer.html to test")
print(f"{'='*80}\n")

# Allow immediate address reuse to avoid TIME_WAIT bind issues when restarting
socketserver.TCPServer.allow_reuse_address = True

# Write PID file so it's easy to find/kill the server if needed
PID_FILE = 'server_phoneme.pid'
try:
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
except Exception:
    pass

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nKeyboard interrupt received, shutting down server')
finally:
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass
