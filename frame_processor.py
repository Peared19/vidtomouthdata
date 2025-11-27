# frame_processor.py
# MediaPipe Face Landmarker Task API - Blend Shapes (52 értékek)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import cv2

def process_frame_full_mouth(image, landmarker):
    """
    Feldolgoz egyetlen képkockát MediaPipe Face Landmarker Task API-val,
    kinyerve a 52 blend shape paramétereit (ML tanítás céljára).
    
    Args:
        image (numpy.ndarray): A feldolgozandó kép (BGR formátumban).
        landmarker: Az előre inicializált MediaPipe FaceLandmarker objektum.

    Returns:
        dict: Egy dictionary 52 blend shape értékekkel, vagy None, ha nem talált arcot.
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

    # ========== BLEND SHAPES (52 érték) ==========
    blend_shape_values = {}
    
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        for blend_shape in result.face_blendshapes[0]:
            blend_shape_values[blend_shape.category_name] = blend_shape.score
    
    # Adatok összegyűjtése a kimenethez
    output_data = {
        "blend_shapes": blend_shape_values
    }

    return output_data