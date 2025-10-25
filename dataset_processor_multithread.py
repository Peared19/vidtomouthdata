import os
import cv2
import dlib
import csv
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from frame_processor import process_frame_full_mouth

# -------------------- Beállítások --------------------
VIDEO_BASE = "D:/MestInt/datasets/gridcorpus/video"
ALIGN_BASE = "D:/MestInt/datasets/gridcorpus/align"
OUTPUT_CSV = "D:/MestInt/datasets/gridcorpus/mouth_data.csv"
TEMP_DIR = "D:/MestInt/datasets/gridcorpus/temp"

os.makedirs("D:/MestInt/datasets/gridcorpus", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# Arcfelismerő és landmark prediktor betöltése
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# -------------------- Segédfüggvény --------------------
def parse_align_file(align_path):
    """
    Betölti az align fájlt és listát ad vissza: [(word, start_time, end_time), ...]
    """
    word_list = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Formátum: start_time end_time word
                word_list.append((parts[2], float(parts[0]), float(parts[1])))
    return word_list

# -------------------- Speaker feldolgozó függvény --------------------
def process_speaker(speaker):
    """
    Feldolgoz egy speakert és a saját temp CSV-jébe írja az adatokat.
    """
    # Minden process betölti a saját detektorát és prediktorát
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    speaker_video_path = os.path.join(VIDEO_BASE, speaker)
    speaker_video_path = os.path.join(speaker_video_path, speaker)

    speaker_align_path = os.path.join(ALIGN_BASE, speaker)
    speaker_align_path = os.path.join(speaker_align_path, "align")

    print(f"[{speaker}] Processing started...")
    print(f"[{speaker}] speaker_video_path: {speaker_video_path}")
    print(f"[{speaker}] speaker_align_path: {speaker_align_path}")

    if not os.path.isdir(speaker_video_path):
        print(f"[{speaker}] ⚠️  Video path not found, skipping...")
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
                print(f"[{speaker}] ⚠️  Missing align file for {video_file}, skipping...")
                continue

            # Betöltjük a transzkripciót
            word_list = parse_align_file(align_path)

            # Videó feldolgozása
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mouth_data = process_frame_full_mouth(frame, detector, predictor)
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
                    json.dumps(mouth_data["inner_lip_relative_points"], separators=(',', ':'))
                ])

                frame_idx += 1

            cap.release()
            print(f"[{speaker}] ✅ Processed {video_file}")
    
    print(f"[{speaker}] ✅ Completed all videos!")
    return speaker

# -------------------- Fő feldolgozás --------------------
if __name__ == "__main__":
    # Speaker-ek listája
    speakers = sorted([s for s in os.listdir(VIDEO_BASE) 
                      if os.path.isdir(os.path.join(VIDEO_BASE, s))])
    
    print(f"🚀 Found {len(speakers)} speakers to process")
    print(f"💻 Using {cpu_count()} CPU cores")
    
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
            "outer_lip_relative_points", "inner_lip_relative_points"
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
                print(f"✅ Merged {speaker}")
    
    # Temp mappa törlése
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass
    
    print("\n🎉 All videos processed and merged successfully!")
