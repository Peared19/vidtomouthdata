import http.server
import socketserver
import json
import torch
import numpy as np
import os
import tempfile
from gtts import gTTS
from pydub import AudioSegment
from inference_simple import load_model, load_vocabulary, predict_blend_shapes

PORT = 8000

# Ensure temp directory exists for audio
AUDIO_DIR = 'temp_audio'
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Load model and vocabulary at startup
print("Initializing server...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_file = 'vocabulary.json'
model_path = 'checkpoints_simple/best_model.pt'
if not os.path.exists(model_path):
    model_path = 'blend_shape_model.pt'

word_to_id = load_vocabulary(vocab_file)
model = load_model(model_path, vocab_size=55, device=device)

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

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/animate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            sentence = data.get('text', '').lower()
            print(f"Received request for: '{sentence}'")
            
            # Simple tokenization (split by space)
            raw_words = sentence.split()
            if not raw_words:
                self.send_response(400)
                self.end_headers()
                return
            
            # Add explicit start/end tokens to the sequence
            words = ['<START>'] + raw_words + ['<END>']

            # 1. Generate Audio per word for exact synchronization
            print("Generating audio per word...")
            combined_audio = AudioSegment.empty()
            word_durations = []
            
            for word in words:
                try:
                    # Handle silence tokens
                    if word.lower() in ['sil', 'sp', '<start>', '<end>']:
                        # Generate 0.5s of silence for explicit pause tokens
                        duration_ms = 500 
                        word_audio = AudioSegment.silent(duration=duration_ms)
                    else:
                        # Generate TTS for single word
                        tts = gTTS(text=word, lang='en', slow=False)
                        
                        # Save to a temporary file
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                            temp_path = f.name
                        
                        # Close the file handle so gTTS can write to it
                        tts.save(temp_path)
                        
                        # Load with pydub
                        word_audio = AudioSegment.from_mp3(temp_path)
                        
                        # Clean up
                        os.remove(temp_path)
                    
                    combined_audio += word_audio
                    word_durations.append(len(word_audio) / 1000.0)
                    
                except Exception as e:
                    print(f"Error generating audio for word '{word}': {e}")
                    word_durations.append(0.5) # Fallback
            
            # Save combined audio
            audio_filename = f"audio_{len(words)}.mp3"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            combined_audio.export(audio_path, format="mp3")
            
            total_duration_sec = len(combined_audio) / 1000.0
            print(f"Total audio duration: {total_duration_sec:.2f}s")
            
            # Generate animation frames
            all_blend_shapes = []
            fps = 30
            
            # Add start/end tokens padding
            # Note: words already contains <START> and <END> at boundaries
            # We need to pad with another layer of <START>/<END> for the window context
            sequence = ['<START>'] + words + ['<END>']
            
            # Generate frames
            for i in range(1, len(sequence) - 1):
                prev_w = sequence[i-1]
                curr_w = sequence[i]
                next_w = sequence[i+1]
                
                prev_id = word_to_id.get(prev_w, 0)
                curr_id = word_to_id.get(curr_w, 0)
                next_id = word_to_id.get(next_w, 0)
                
                # Get duration for this specific word
                duration = word_durations[i-1]
                
                frames_per_word = int(duration * fps)
                if frames_per_word < 1: frames_per_word = 1
                
                # Generate frames for this word
                for f in range(frames_per_word):
                    frame_pos = f / frames_per_word
                    
                    # Check for silence/special tokens
                    if curr_w.lower() in ['sil', 'sp', '<start>', '<end>']:
                        # Force neutral face (all zeros)
                        blend_shapes = np.zeros(52, dtype=np.float32)
                    else:
                        # Predict
                        blend_shapes = predict_blend_shapes(
                            model, prev_id, curr_id, next_id, 
                            frame_pos, duration, device
                        )
                    
                    all_blend_shapes.append(blend_shapes)
            
            # Convert to numpy and smooth
            all_blend_shapes = np.array(all_blend_shapes)
            if len(all_blend_shapes) > 0:
                # Apply smoothing to fix snapping between words
                all_blend_shapes = smooth_data(all_blend_shapes, window_size=5)
            
            # Convert back to dictionary format
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
                'audio_url': f"/{AUDIO_DIR}/{audio_filename}"
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            print(f"Generated {len(frames)} frames")
        else:
            # Serve static files (including audio)
            super().do_GET()

    def do_GET(self):
        # Allow serving files from the temp audio directory
        if self.path.startswith(f"/{AUDIO_DIR}/"):
            # Simple security check to prevent directory traversal
            if ".." in self.path:
                self.send_error(403)
                return
            super().do_GET()
        else:
            super().do_GET()

print(f"Serving at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
