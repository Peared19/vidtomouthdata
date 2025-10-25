# frame_processor.py

import dlib
import numpy as np
import cv2 # A szürkeárnyalatosításhoz kell

# Ezek konstansok, definiálhatjuk a függvényen kívül
MOUTH_OUTER_POINTS_INDICES = list(range(48, 60))
MOUTH_INNER_POINTS_INDICES = list(range(60, 68))

def process_frame_full_mouth(image, detector, predictor):
    """
    Feldolgoz egyetlen képkockát, kinyerve a külső és belső ajak pontjait.
    Args:
        image (numpy.ndarray): A feldolgozandó kép (színes, BGR formátumban).
        detector: Az előre inicializált dlib arcfelismerő.
        predictor: Az előre inicializált dlib landmark prediktor.

    Returns:
        dict: Egy dictionary a teljes száj adataival, vagy None, ha nem talált arcot.
    """
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(grayimg)

    if len(faces) == 0:
        return None 

    #  első detektált arccal dolgozunk
    face = faces[0]
    
    landmarks = predictor(grayimg, face)

    landmark_array = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Külső és belső ajak pontjainak kinyerése
    outer_mouth_coords = landmark_array[MOUTH_OUTER_POINTS_INDICES]
    inner_mouth_coords = landmark_array[MOUTH_INNER_POINTS_INDICES]

    # középpont számítása
    all_mouth_coords = np.concatenate([outer_mouth_coords, inner_mouth_coords])
    mouth_center = np.mean(all_mouth_coords, axis=0).astype(int)

    # Relatív pozíciók kiszámítása
    relative_outer_mouth_points = outer_mouth_coords - mouth_center
    relative_inner_mouth_points = inner_mouth_coords - mouth_center

    # Adatok összegyűjtése a kimenethez
    output_data = {
        "mouth_center": mouth_center.tolist(),
        "outer_lip_indices": MOUTH_OUTER_POINTS_INDICES,
        "outer_lip_relative_points": relative_outer_mouth_points.tolist(),
        "inner_lip_indices": MOUTH_INNER_POINTS_INDICES,
        "inner_lip_relative_points": relative_inner_mouth_points.tolist()
    }
    
    return output_data