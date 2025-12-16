import os
import cv2
import csv
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Pool, cpu_count
from frame_processor import process_frame_full_mouth
from g2p_en import G2p
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# -------------------- Settings --------------------
VIDEO_BASE = "gridcorpus/video"
ALIGN_BASE = "gridcorpus/align"
OUTPUT_CSV = "gridcorpus/mouth_data_phonemes.csv"
TEMP_DIR = "gridcorpus/temp_phonemes"
MODEL_PATH = "face_landmarker.task"

os.makedirs("gridcorpus", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Download Face Landmarker if needed
if not os.path.exists(MODEL_PATH):
    print("⏬ Downloading Face Landmarker model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded!")
    except Exception as e:
        print(f"Model download error: {e}")
        exit(1)

def get_phonemes_for_word(g2p, word):
    """
    Converts a word to a list of phonemes using g2p_en.
    Handles special tokens from Grid Corpus like 'sil', 'sp'.
    """
    if word in ['sil', 'sp']:
        return [word]
    
    # g2p expects a string.
    phonemes = g2p(word)
    # Filter out empty strings or non-phoneme chars if any
    phonemes = [p for p in phonemes if p.strip() != '']
    
    # If g2p returns nothing (e.g. for punctuation), return 'sp' or similar?
    # Grid corpus words are usually clean.
    if not phonemes:
        return ['sp']
        
    return phonemes

def parse_align_file_to_phonemes(align_path, g2p, sample_rate=25000):
    """
    Parses align file and converts words to phonemes with uniform duration.
    Returns: [(phoneme, start_time_s, end_time_s), ...]
    """
    phoneme_list = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_sample = float(parts[0])
                end_sample = float(parts[1])
                word = parts[2]
                
                start_time_s = start_sample / sample_rate
                end_time_s = end_sample / sample_rate
                duration = end_time_s - start_time_s
                
                phonemes = get_phonemes_for_word(g2p, word)
                
                if not phonemes:
                    continue
                    
                # Uniform Duration Approximation
                # Divide duration equally among phonemes
                phoneme_duration = duration / len(phonemes)
                
                for i, p in enumerate(phonemes):
                    p_start = start_time_s + (i * phoneme_duration)
                    p_end = start_time_s + ((i + 1) * phoneme_duration)
                    phoneme_list.append((p, p_start, p_end))
                    
    return phoneme_list

def process_speaker(speaker):
    # Initialize G2P inside the process
    g2p = G2p()
    
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True
    )
    
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    speaker_video_path = os.path.join(VIDEO_BASE, speaker)
    speaker_video_path = os.path.join(speaker_video_path, speaker)

    speaker_align_path = os.path.join(ALIGN_BASE, speaker)
    speaker_align_path = os.path.join(speaker_align_path, "align")

    print(f"[{speaker}] Processing started...")

    if not os.path.isdir(speaker_video_path):
        print(f"[{speaker}] Video path not found, skipping...")
        return

    temp_csv = os.path.join(TEMP_DIR, f"{speaker}.csv")
    
    with open(temp_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        
        for video_file in sorted(os.listdir(speaker_video_path)):
            if not video_file.lower().endswith((".mpg", ".mp4")):
                continue

            video_path = os.path.join(speaker_video_path, video_file)
            align_file_name = os.path.splitext(video_file)[0] + ".align"
            align_path = os.path.join(speaker_align_path, align_file_name)

            if not os.path.exists(align_path):
                print(f"[{speaker}] Missing align file for {video_file}, skipping...")
                continue

            # Load phoneme alignments
            phoneme_list = parse_align_file_to_phonemes(align_path, g2p, sample_rate=25000)
            
            # Phoneme list for context
            phonemes_in_order = [p for p, _, _ in phoneme_list]

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            
            # Tracking frame count per phoneme instance
            # Since phonemes can repeat, we need to track by index in the list
            frame_count_in_phoneme = {} # phoneme_idx -> count

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mouth_data = process_frame_full_mouth(frame, landmarker)
                if mouth_data is None:
                    frame_idx += 1
                    continue

                current_time = frame_idx / fps
                phoneme_for_frame = None
                phoneme_idx = None
                
                # Find which phoneme is active
                for idx, (phoneme, start_time, end_time) in enumerate(phoneme_list):
                    if start_time <= current_time <= end_time:
                        phoneme_for_frame = phoneme
                        phoneme_idx = idx
                        break

                if phoneme_for_frame is None:
                    frame_idx += 1
                    continue

                if phoneme_idx not in frame_count_in_phoneme:
                    frame_count_in_phoneme[phoneme_idx] = 0
                
                frame_pos_in_phoneme = frame_count_in_phoneme[phoneme_idx]
                frame_count_in_phoneme[phoneme_idx] += 1
                
                # Context: Prev and Next Phoneme
                prev_phoneme = phonemes_in_order[phoneme_idx - 1] if phoneme_idx > 0 else "<START>"
                next_phoneme = phonemes_in_order[phoneme_idx + 1] if phoneme_idx < len(phonemes_in_order) - 1 else "<END>"
                
                # Phoneme duration in frames
                _, start_time, end_time = phoneme_list[phoneme_idx]
                phoneme_duration_frames = int((end_time - start_time) * fps) + 1
                
                # Relative position 0.0-1.0
                frame_pos = frame_pos_in_phoneme / max(phoneme_duration_frames, 1)

                # Write to CSV
                writer.writerow([
                    speaker,
                    video_file,
                    frame_idx,
                    prev_phoneme,
                    phoneme_for_frame,
                    next_phoneme,
                    round(frame_pos, 4),
                    phoneme_duration_frames,
                    json.dumps(mouth_data["blend_shapes"], separators=(',', ':'))
                ])

                frame_idx += 1

            cap.release()
            # print(f"[{speaker}]  Processed {video_file}")
    
    landmarker.close()
    print(f"[{speaker}] Completed all videos!")
    return speaker

if __name__ == "__main__":
    speakers = sorted([s for s in os.listdir(VIDEO_BASE) 
                      if os.path.isdir(os.path.join(VIDEO_BASE, s))])
    
    print(f"Found {len(speakers)} speakers to process")
    print(f"Using {cpu_count()} CPU cores")
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_speaker, speakers)
    
    print("\nMerging all temporary CSV files...")
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=';')
        
        writer.writerow([
            "speaker", "video", "frame_idx", 
            "prev_phoneme", "curr_phoneme", "next_phoneme", 
            "frame_pos", "phoneme_duration_frames", 
            "blend_shapes"
        ])
        
        for speaker in speakers:
            temp_csv = os.path.join(TEMP_DIR, f"{speaker}.csv")
            if os.path.exists(temp_csv):
                with open(temp_csv, "r", encoding="utf-8") as infile:
                    reader = csv.reader(infile, delimiter=';')
                    for row in reader:
                        writer.writerow(row)
                os.remove(temp_csv)
                print(f"Merged {speaker}")
    
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass
    print("Done! Saved to", OUTPUT_CSV)
