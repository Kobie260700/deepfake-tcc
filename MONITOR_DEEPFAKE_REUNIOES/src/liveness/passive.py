from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LivenessResult:
    temporal_variation: float
    landmark_movement: float
    valid: bool
    confidence: float


def normalized_landmarks(face) -> np.ndarray:
    x1, y1, x2, y2 = face.bbox.astype(float)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    points = np.asarray(face.kps, dtype=np.float32).copy()
    points[:, 0] = (points[:, 0] - x1) / width
    points[:, 1] = (points[:, 1] - y1) / height
    return points


def normalized_face_crop(frame: np.ndarray, face) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.round(face.bbox).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (128, 128))


def analyze_liveness(
    landmarks: list[np.ndarray],
    crops: list[np.ndarray],
    minimum_variation: float,
    minimum_movement: float,
    maximum_movement: float,
) -> LivenessResult:
    movements = [
        float(np.mean(np.linalg.norm(current - previous, axis=1)))
        for previous, current in zip(landmarks, landmarks[1:])
    ]
    variations = [
        float(np.mean(cv2.absdiff(previous, current)))
        for previous, current in zip(crops, crops[1:])
    ]
    movement = float(np.mean(movements)) if movements else 0.0
    variation = float(np.mean(variations)) if variations else 0.0
    valid = minimum_variation <= variation and minimum_movement <= movement <= maximum_movement
    variation_score = np.clip(variation / max(minimum_variation * 3, 1e-6), 0.0, 1.0)
    if movement < minimum_movement:
        movement_score = np.clip(movement / max(minimum_movement, 1e-6), 0.0, 1.0)
    elif movement <= maximum_movement:
        movement_score = 1.0
    else:
        movement_score = np.clip(maximum_movement / max(movement, 1e-6), 0.0, 1.0)
    confidence = float((variation_score + movement_score) / 2)
    if not valid:
        confidence = min(confidence, 0.49)
    return LivenessResult(variation, movement, valid, confidence)
