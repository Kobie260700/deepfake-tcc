import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.face_tracking.analyzer import FaceMonitorAnalyzer


class FaceSelectionTests(unittest.TestCase):
    def test_selects_face_closest_to_reference(self):
        analyzer = FaceMonitorAnalyzer.__new__(FaceMonitorAnalyzer)
        far = SimpleNamespace(embedding=np.full(4, 10.0, dtype=np.float32))
        close = SimpleNamespace(embedding=np.full(4, 1.0, dtype=np.float32))
        analyzer.detect_faces = Mock(return_value=[far, close])

        selected = analyzer.select_identity_face(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros(4, dtype=np.float32),
        )

        self.assertIs(selected, close)


if __name__ == "__main__":
    unittest.main()
