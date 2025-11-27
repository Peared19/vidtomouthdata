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

# -------------------- Beállítások --------------------
VIDEO_BASE = "gridcorpus/video"
ALIGN_BASE = "gridcorpus/align"
OUTPUT_CSV = "gridcorpus/mouth_data_context.csv"
TEMP_DIR = "gridcorpus/temp"
MODEL_PATH = "face_landmarker.task"

os.makedirs("gridcorpus", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Face Landmarker model letöltése ha nincs meg
if not os.path.exists(MODEL_PATH):
    print("⏬ Face Landmarker model letöltése...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model letöltve!")
    except Exception as e:
        print(f"Model letöltés hiba: {e}")
        print("Kérjük, töltse le kézzel innen:")
        print(url)
        exit(1)

# -------------------- Segédfüggvény --------------------
def parse_align_file(align_path, sample_rate=25000):
    """
    Betölti az align fájlt és listát ad vissza: [(word, start_time_s, end_time_s), ...]
    Az align fájlban a GRID corpus mintaszámokat tartalmaz (nem másodperceket),
    ezért konvertálni kell a sample_rate alapján.
    """
    word_list = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_sample = float(parts[0])
                end_sample = float(parts[1])
                word = parts[2]
                # Átváltás másodpercre:
                start_time_s = start_sample / sample_rate
                end_time_s = end_sample / sample_rate
                word_list.append((word, start_time_s, end_time_s))
    return word_list

# -------------------- Speaker feldolgozó függvény --------------------
def process_speaker(speaker):
    """
    Feldolgoz egy speakert és a saját temp CSV-jébe írja az adatokat.
    Frame-szintű context-aware formátumban (prev_word, curr_word, next_word, frame_pos).
    """
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Minden process saját FaceLandmarker objektumot hoz létre
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
    print(f"[{speaker}] speaker_video_path: {speaker_video_path}")
    print(f"[{speaker}] speaker_align_path: {speaker_align_path}")

    if not os.path.isdir(speaker_video_path):
        print(f"[{speaker}] Video path not found, skipping...")
        return

    # Ideiglenes CSV fájl ehhez a speakerhez
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

            # Betöltjük a transzkripciót
            word_list = parse_align_file(align_path, sample_rate=25000)
            
            # Szósorrendet kigyűjtjük (csak a szavak listája)
            words_in_order = [word for word, _, _ in word_list]

            # Videó feldolgozása
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            
            # Nyomkövetés: mely szó van most
            frame_count_in_word = {}  # word_idx → frame count

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mouth_data = process_frame_full_mouth(frame, landmarker)
                if mouth_data is None:
                    frame_idx += 1
                    continue

                # Szó meghatározása az aktuális frame idő alapján
                current_time = frame_idx / fps
                word_for_frame = None
                word_idx = None
                
                for idx, (word, start_time, end_time) in enumerate(word_list):
                    if start_time <= current_time <= end_time:
                        word_for_frame = word
                        word_idx = idx
                        break

                if word_for_frame is None:
                    frame_idx += 1
                    continue

                # Frame számlálása az aktuális szóban
                if word_idx not in frame_count_in_word:
                    frame_count_in_word[word_idx] = 0
                
                frame_pos_in_word = frame_count_in_word[word_idx]
                frame_count_in_word[word_idx] += 1
                
                # Prev és next szó
                prev_word = words_in_order[word_idx - 1] if word_idx > 0 else "<START>"
                next_word = words_in_order[word_idx + 1] if word_idx < len(words_in_order) - 1 else "<END>"
                
                # Szó teljes frame száma (hozzávetőleges)
                # Az align fájlból: end_time - start_time
                _, start_time, end_time = word_list[word_idx]
                word_duration_frames = int((end_time - start_time) * fps) + 1
                
                # Relatív pozíció 0.0-1.0 között
                frame_pos = frame_pos_in_word / max(word_duration_frames, 1)

                # Mentés CSV-be (context-aware formát)
                writer.writerow([
                    speaker,
                    video_file,
                    frame_idx,
                    prev_word if prev_word else "None",
                    word_for_frame,
                    next_word if next_word else "None",
                    round(frame_pos, 4),  # Relatív pozíció 0.0-1.0
                    word_duration_frames,  # A szó teljes hossza frame-ben
                    json.dumps(mouth_data["blend_shapes"], separators=(',', ':'))
                ])

                frame_idx += 1

            cap.release()
            print(f"[{speaker}]  Processed {video_file}")
    
    # FaceLandmarker felszabadítása
    landmarker.close()
    
    print(f"[{speaker}] Completed all videos!")
    return speaker

# -------------------- Fő feldolgozás --------------------
if __name__ == "__main__":
    # Speaker-ek listája
    speakers = sorted([s for s in os.listdir(VIDEO_BASE) 
                      if os.path.isdir(os.path.join(VIDEO_BASE, s))])
    
    print(f"Found {len(speakers)} speakers to process")
    print(f"Using {cpu_count()} CPU cores")
    
    # Párhuzamos feldolgozás
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_speaker, speakers)
    
    print("\nMerging all temporary CSV files...")
    
    # Összefűzzük az ideiglenes CSV-ket
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=';')
        
        # Fejléc írása (frame-szintű context-aware formátum)
        writer.writerow([
            "speaker", "video", "frame_idx", 
            "prev_word", "curr_word", "next_word", 
            "frame_pos", "word_duration_frames", 
            "blend_shapes"
        ])
        
        # Minden speaker temp CSV-jét beolvassuk
        for speaker in speakers:
            temp_csv = os.path.join(TEMP_DIR, f"{speaker}.csv")
            if os.path.exists(temp_csv):
                with open(temp_csv, "r", encoding="utf-8") as infile:
                    reader = csv.reader(infile, delimiter=';')
                    for row in reader:
                        writer.writerow(row)
                # Töröljük a temp fájlt
                os.remove(temp_csv)
                print(f"Merged {speaker}")
    
    # Temp mappa törlése
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass
    

