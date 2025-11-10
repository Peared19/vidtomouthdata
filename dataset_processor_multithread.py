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
VIDEO_BASE = "D:/MestInt/datasets/gridcorpus/video"
ALIGN_BASE = "D:/MestInt/datasets/gridcorpus/align"
OUTPUT_CSV = "D:/MestInt/word_tomoutmap/mouth_data.csv"
TEMP_DIR = "D:/MestInt/word_tomoutmap/temp"
MODEL_PATH = "face_landmarker.task"

os.makedirs("D:/MestInt/datasets/gridcorpus", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Face Landmarker model letöltése ha nincs meg
if not os.path.exists(MODEL_PATH):
    print("⏬ Face Landmarker model letöltése...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅ Model letöltve!")
    except Exception as e:
        print(f"❌ Model letöltés hiba: {e}")
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

            # Videó feldolgozása
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

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
                for word, start_time, end_time in word_list:
                    if start_time <= current_time <= end_time:
                        word_for_frame = word
                        break

                if word_for_frame is None:
                    frame_idx += 1
                    continue

                # Mentés CSV-be
                writer.writerow([
                    speaker,
                    video_file,
                    frame_idx,
                    word_for_frame,
                    mouth_data["mouth_center"][0],
                    mouth_data["mouth_center"][1],
                    json.dumps(mouth_data["outer_lip_relative_points"], separators=(',', ':')),
                    json.dumps(mouth_data["inner_lip_relative_points"], separators=(',', ':')),
                    json.dumps(mouth_data["blend_shapes"], separators=(',', ':')),
                    json.dumps(mouth_data["mouth_blend_shapes"], separators=(',', ':')),
                    json.dumps(mouth_data["eyes_blend_shapes"], separators=(',', ':')),
                    json.dumps(mouth_data["brow_blend_shapes"], separators=(',', ':')),
                    json.dumps(mouth_data["face_shape_blend_shapes"], separators=(',', ':')),
                    json.dumps(mouth_data["3d_landmarks"], separators=(',', ':')),
                    json.dumps(mouth_data["pixel_landmarks"], separators=(',', ':')),
                    json.dumps(mouth_data["relative_landmarks"], separators=(',', ':')),
                    json.dumps(mouth_data["face_center_pixel"], separators=(',', ':')),
                    json.dumps(mouth_data["face_center_3d"], separators=(',', ':'))
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
    
    print("\n🔗 Merging all temporary CSV files...")
    
    # Összefűzzük az ideiglenes CSV-ket
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=';')
        
        # Fejléc írása
        writer.writerow([
            "speaker", "video", "frame_idx", "word",
            "mouth_center_x", "mouth_center_y",
            "outer_lip_relative_points", "inner_lip_relative_points",
            "blend_shapes", "mouth_blend_shapes",
            "eyes_blend_shapes", "brow_blend_shapes", "face_shape_blend_shapes",
            "3d_landmarks", "pixel_landmarks", "relative_landmarks",
            "face_center_pixel", "face_center_3d"
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
    

