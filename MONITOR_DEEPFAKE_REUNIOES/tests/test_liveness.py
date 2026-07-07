import unittest

import numpy as np

from src.liveness.passive import analyze_liveness


class LivenessTests(unittest.TestCase):
    def test_detects_temporal_activity(self):
        landmarks = [
            np.zeros((5, 2), dtype=np.float32),
            np.full((5, 2), 0.01, dtype=np.float32),
            np.full((5, 2), 0.02, dtype=np.float32),
        ]
        crops = [
            np.zeros((16, 16), dtype=np.uint8),
            np.full((16, 16), 10, dtype=np.uint8),
            np.full((16, 16), 20, dtype=np.uint8),
        ]
        result = analyze_liveness(landmarks, crops, 1.0, 0.0005, 0.08)
        self.assertTrue(result.valid)
        self.assertGreater(result.temporal_variation, 1.0)

    def test_static_window_is_inconclusive(self):
        landmarks = [np.zeros((5, 2), dtype=np.float32)] * 3
        crops = [np.zeros((16, 16), dtype=np.uint8)] * 3
        result = analyze_liveness(landmarks, crops, 1.0, 0.0005, 0.08)
        self.assertFalse(result.valid)

    def test_excessive_movement_has_low_confidence(self):
        landmarks = [
            np.zeros((5, 2), dtype=np.float32),
            np.full((5, 2), 0.5, dtype=np.float32),
        ]
        crops = [
            np.zeros((16, 16), dtype=np.uint8),
            np.full((16, 16), 20, dtype=np.uint8),
        ]
        result = analyze_liveness(landmarks, crops, 1.0, 0.0005, 0.08)
        self.assertFalse(result.valid)
        self.assertLess(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
