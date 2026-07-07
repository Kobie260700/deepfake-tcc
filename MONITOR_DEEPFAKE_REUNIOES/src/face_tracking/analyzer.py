from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from src.config import MonitorConfig


@dataclass(frozen=True)
class FaceQuality:
    adequate: bool
    detection_score: float
    face_size: int
    brightness: float
    sharpness: float
    reasons: tuple[str, ...]


class FaceMonitorAnalyzer:
    def __init__(self, config: MonitorConfig):
        self.config = config
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition", "landmark_3d_68"],
        )
        self.app.prepare(
            ctx_id=0,
            det_size=(config.detector_input_size, config.detector_input_size),
        )

    def detect_primary_face(self, frame: np.ndarray) -> Any | None:
        faces = self.detect_faces(frame)
        if not faces:
            return None
        return max(faces, key=lambda face: self._bbox_area(face.bbox))

    def detect_faces(self, frame: np.ndarray) -> list[Any]:
        return list(self.app.get(frame))

    def select_identity_face(
        self,
        frame: np.ndarray,
        reference_embedding: np.ndarray,
    ) -> Any | None:
        faces = self.detect_faces(frame)
        if not faces:
            return None
        return min(
            faces,
            key=lambda face: float(
                np.linalg.norm(reference_embedding - self.embedding(face))
            ),
        )

    def evaluate_quality(
        self,
        frame: np.ndarray,
        face: Any,
        minimum_sharpness: float | None = None,
    ) -> FaceQuality:
        x1, y1, x2, y2 = self._clamped_bbox(frame, face.bbox)
        crop = frame[y1:y2, x1:x2]
        detection_score = float(face.det_score)
        face_size = int(min(x2 - x1, y2 - y1))

        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            brightness = 0.0
            sharpness = 0.0

        reasons = []
        if face_size < self.config.minimum_face_size:
            reasons.append("rosto_pequeno")
        if detection_score < self.config.minimum_detection_score:
            reasons.append("baixa_confianca")
        if not self.config.minimum_brightness <= brightness <= self.config.maximum_brightness:
            reasons.append("iluminacao_inadequada")
        sharpness_threshold = (
            self.config.minimum_sharpness
            if minimum_sharpness is None
            else minimum_sharpness
        )
        if sharpness < sharpness_threshold:
            reasons.append("baixa_nitidez")

        return FaceQuality(
            adequate=not reasons,
            detection_score=detection_score,
            face_size=face_size,
            brightness=brightness,
            sharpness=sharpness,
            reasons=tuple(reasons),
        )

    @staticmethod
    def embedding(face: Any) -> np.ndarray:
        return np.asarray(face.embedding, dtype=np.float32)

    @staticmethod
    def face_crop(frame: np.ndarray, face: Any, margin_ratio: float = 0.15) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(float)
        margin_x = (x2 - x1) * margin_ratio
        margin_y = (y2 - y1) * margin_ratio
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(width, int(x2 + margin_x))
        y2 = min(height, int(y2 + margin_y))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def aligned_face_crop(frame: np.ndarray, face: Any, output_size: int = 224) -> np.ndarray | None:
        """Align a face to the five-point geometry used by DeepfakeBench."""
        keypoints = getattr(face, "kps", None)
        if keypoints is None or np.asarray(keypoints).shape != (5, 2):
            return None
        source = np.asarray(keypoints, dtype=np.float32)
        destination = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        destination *= output_size / 112.0
        # The official Effort demo uses scale=1.3, leaving context around the face.
        destination = (destination + output_size * 0.15) / 1.3
        transform, _ = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.LMEDS,
        )
        if transform is None:
            return None
        return cv2.warpAffine(
            frame,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(x2 - x1, 0) * max(y2 - y1, 0))

    @staticmethod
    def _clamped_bbox(frame: np.ndarray, bbox: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.round(bbox).astype(int)
        return (
            max(0, min(x1, width)),
            max(0, min(y1, height)),
            max(0, min(x2, width)),
            max(0, min(y2, height)),
        )


def draw_face_analysis(frame: np.ndarray, face: Any, quality: FaceQuality) -> None:
    x1, y1, x2, y2 = np.round(face.bbox).astype(int)
    color = (0, 200, 0) if quality.adequate else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if getattr(face, "kps", None) is not None:
        for point in np.round(face.kps).astype(int):
            cv2.circle(frame, tuple(point), 2, color, -1, cv2.LINE_AA)

    status = "QUALIDADE ADEQUADA" if quality.adequate else "QUALIDADE INSUFICIENTE"
    cv2.putText(frame, status, (max(x1, 5), max(y1 - 12, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
