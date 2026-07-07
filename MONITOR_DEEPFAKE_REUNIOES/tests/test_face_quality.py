import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.config import MonitorConfig
from src.face_tracking.analyzer import FaceMonitorAnalyzer


class FaceQualityTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = FaceMonitorAnalyzer.__new__(FaceMonitorAnalyzer)
        self.analyzer.config = MonitorConfig(
            minimum_face_size=80,
            minimum_detection_score=0.5,
            minimum_brightness=35.0,
            maximum_brightness=220.0,
            minimum_sharpness=30.0,
        )

    def test_accepts_well_lit_sharp_face_crop(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        checkerboard = np.indices((120, 120)).sum(axis=0) % 2
        crop = (checkerboard * 140 + 50).astype(np.uint8)
        frame[60:180, 60:180] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        face = SimpleNamespace(bbox=np.array([60, 60, 180, 180]), det_score=0.95)

        quality = self.analyzer.evaluate_quality(frame, face)

        self.assertTrue(quality.adequate)
        self.assertEqual(quality.reasons, ())

    def test_rejects_small_dark_blurred_face_crop(self):
        frame = np.full((160, 160, 3), 10, dtype=np.uint8)
        face = SimpleNamespace(bbox=np.array([20, 20, 70, 70]), det_score=0.4)

        quality = self.analyzer.evaluate_quality(frame, face)

        self.assertFalse(quality.adequate)
        self.assertIn("rosto_pequeno", quality.reasons)
        self.assertIn("baixa_confianca", quality.reasons)
        self.assertIn("iluminacao_inadequada", quality.reasons)
        self.assertIn("baixa_nitidez", quality.reasons)

    def test_monitor_can_use_relaxed_sharpness_for_compressed_video(self):
        frame = np.full((240, 240, 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (60, 60), (180, 180), (115, 115, 115), 2)
        face = SimpleNamespace(bbox=np.array([60, 60, 180, 180]), det_score=0.95)

        strict = self.analyzer.evaluate_quality(frame, face)
        relaxed = self.analyzer.evaluate_quality(frame, face, minimum_sharpness=1.0)

        self.assertIn("baixa_nitidez", strict.reasons)
        self.assertNotIn("baixa_nitidez", relaxed.reasons)

    def test_aligns_face_from_five_keypoints(self):
        frame = np.full((300, 300, 3), 120, dtype=np.uint8)
        face = SimpleNamespace(
            kps=np.array(
                [[100, 120], [180, 118], [142, 160], [112, 200], [174, 198]],
                dtype=np.float32,
            )
        )

        crop = self.analyzer.aligned_face_crop(frame, face)

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
