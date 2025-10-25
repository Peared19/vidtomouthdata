import os
import cv2
import dlib
import csv
import json
import numpy as np
from frame_processor import process_frame_full_mouth

# -------------------- Beállítások --------------------
VIDEO_BASE = "D:/MestInt/datasets/gridcorpus/video"
ALIGN_BASE = "D:/MestInt/datasets/gridcorpus/align"
OUTPUT_CSV = "D:/MestInt/datasets/gridcorpus/mouth_data.csv"

os.makedirs("D:/MestInt/datasets/gridcorpus", exist_ok=True)
OUTPUT_CSV = "D:/MestInt/datasets/gridcorpus/mouth_data.csv"


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

# -------------------- Fő feldolgozás --------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    # Fejléc
    writer.writerow([
        "speaker", "video", "frame_idx", "word",
        "mouth_center_x", "mouth_center_y",
        "outer_lip_relative_points", "inner_lip_relative_points"
    ])

    # Minden speaker mappa
    for speaker in sorted(os.listdir(VIDEO_BASE)):
        speaker_video_path = os.path.join(VIDEO_BASE, speaker)
        speaker_video_path = os.path.join(speaker_video_path, speaker)

        speaker_align_path = os.path.join(ALIGN_BASE, speaker)
        speaker_align_path = os.path.join(speaker_align_path, "align")
        


        print(f"speaker_video_path: {speaker_video_path}")
        print(f"speaker_align_path: {speaker_align_path}")

        if not os.path.isdir(speaker_video_path):
            continue

        for video_file in sorted(os.listdir(speaker_video_path)):
            if not video_file.lower().endswith((".mpg", ".mp4")):
                continue

            video_path = os.path.join(speaker_video_path, video_file)
            align_file_name = os.path.splitext(video_file)[0] + ".align"
            align_path = os.path.join(speaker_align_path, align_file_name)

            if not os.path.exists(align_path):
                print(f"⚠️  Missing align file for {video_file}, skipping...")
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
                    json.dumps(mouth_data["outer_lip_relative_points"]),
                    json.dumps(mouth_data["inner_lip_relative_points"])
                ])

                frame_idx += 1

            cap.release()
            print(f"✅ Processed {video_file} for {speaker}")

print("\n🎉 All videos processed and saved to CSV successfully!")
