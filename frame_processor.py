# frame_processor.py
# MediaPipe Face Landmarker Task API - Blend Shapes támogatás

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import cv2

# Mouth landmark indices (MediaPipe Face Mesh)
MOUTH_OUTER_POINTS_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 17, 181, 91, 146
]
MOUTH_INNER_POINTS_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 14, 178, 88, 95
]

def process_frame_full_mouth(image, landmarker):
    """
    Feldolgoz egyetlen képkockát MediaPipe Face Landmarker Task API-val,
    kinyerve a teljes 3D arc modell adatait és blend shape paramétereit.
    
    Args:
        image (numpy.ndarray): A feldolgozandó kép (BGR formátumban).
        landmarker: Az előre inicializált MediaPipe FaceLandmarker objektum.

    Returns:
        dict: Egy dictionary a száj adataival és blend shape paramétereivel, vagy None, ha nem talált arcot.
    """
    # Kép konvertálása MediaPipe Image objektummá
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
    
    # Feldolgozás
    try:
        result = landmarker.detect(mp_image)
    except Exception as e:
        print(f"Hiba a face detection során: {e}")
        return None

    if not result.face_landmarks:
        return None

    # Az első detektált arccal dolgozunk
    landmarks = result.face_landmarks[0]
    landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Képpont koordináták (pixel értékek)
    image_height, image_width = image.shape[:2]
    pixel_coords = landmark_array[:, :2] * np.array([image_width, image_height])

    # Külső és belső ajak pontjainak kinyerése
    outer_mouth_coords = pixel_coords[MOUTH_OUTER_POINTS_INDICES]
    inner_mouth_coords = pixel_coords[MOUTH_INNER_POINTS_INDICES]

    # Szájközéppont számítása
    all_mouth_coords = np.concatenate([outer_mouth_coords, inner_mouth_coords])
    mouth_center = np.mean(all_mouth_coords, axis=0).astype(int)

    # Relatív pozíciók kiszámítása
    relative_outer_mouth_points = outer_mouth_coords - mouth_center
    relative_inner_mouth_points = inner_mouth_coords - mouth_center

    # ========== BLEND SHAPES ==========
    blend_shape_values = {}
    
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        for blend_shape in result.face_blendshapes[0]:
            blend_shape_values[blend_shape.category_name] = blend_shape.score
    
    # Szájmozgási specifikus blend shape-ek
    mouth_blend_shapes = {
        key: blend_shape_values.get(key, 0.0)
        for key in ['mouthOpen', 'mouthRight', 'mouthLeft', 'mouthFunnel', 
                    'mouthPucker', 'jawOpen', 'mouthClose', 'mouthSmileLeft', 
                    'mouthSmileRight', 'mouthUpperUpLeft', 'mouthUpperUpRight']
    }
    
    # Szemek blend shapes
    eyes_blend_shapes = {
        key: blend_shape_values.get(key, 0.0)
        for key in ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookUpLeft', 'eyeLookUpRight',
                    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
                    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeWideLeft', 'eyeWideRight',
                    'eyeSquintLeft', 'eyeSquintRight']
    }
    
    # Szemöldök blend shapes
    brow_blend_shapes = {
        key: blend_shape_values.get(key, 0.0)
        for key in ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']
    }
    
    # Arc formája (arccsontok, orcák, stb.)
    face_shape_blend_shapes = {
        key: blend_shape_values.get(key, 0.0)
        for key in ['cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'cheekRaiseLeft', 'cheekRaiseRight',
                    'noseSneerLeft', 'noseSneerRight', 'jawForward', 'jawLeft', 'jawRight']
    }
    
    # Normalizálás az arc középpontjához (arc centroidja)
    face_center = np.mean(landmark_array, axis=0)
    face_center_pixel = face_center[:2] * np.array([image_width, image_height])
    
    # Összes landmark relatív pozíciói az arc központjához képest
    relative_landmarks_pixel = pixel_coords - face_center_pixel

    # Adatok összegyűjtése a kimenethez
    output_data = {
        # ========== SZÁJ SPECIFIKUS ADATOK ==========
        "mouth_center": mouth_center.tolist(),
        "mouth_center_3d": np.mean(landmark_array[MOUTH_OUTER_POINTS_INDICES + MOUTH_INNER_POINTS_INDICES], axis=0).tolist(),
        "outer_lip_pixel_points": outer_mouth_coords.tolist(),
        "outer_lip_relative_points": relative_outer_mouth_points.tolist(),
        "inner_lip_pixel_points": inner_mouth_coords.tolist(),
        "inner_lip_relative_points": relative_inner_mouth_points.tolist(),
        
        # ========== BLEND SHAPES (ÖSSZES) ==========
        "blend_shapes": blend_shape_values,
        "mouth_blend_shapes": mouth_blend_shapes,
        "eyes_blend_shapes": eyes_blend_shapes,
        "brow_blend_shapes": brow_blend_shapes,
        "face_shape_blend_shapes": face_shape_blend_shapes,
        
        # ========== TELJES ARC MODELL ==========
        "3d_landmarks": landmark_array.tolist(),  # 478 x 3D pont
        "pixel_landmarks": pixel_coords.tolist(),  # 478 x 2D pont (pixel koordináták)
        "relative_landmarks": relative_landmarks_pixel.tolist(),  # 478 pont az arc centrumhoz képest normalizálva
        "face_center_pixel": face_center_pixel.tolist(),
        "face_center_3d": face_center.tolist()
    }

    return output_data