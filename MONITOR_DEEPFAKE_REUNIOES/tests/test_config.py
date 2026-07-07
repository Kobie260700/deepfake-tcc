import json
import tempfile
import unittest
from pathlib import Path

from src.config import MonitorConfig
from src.monitor import region_from_args


class MonitorConfigTests(unittest.TestCase):
    def test_loads_configuration_file(self):
        values = {
            "sample_interval_seconds": 5.0,
            "target_capture_fps": 8.0,
            "detector_input_size": 320,
            "minimum_face_size": 64,
            "minimum_detection_score": 0.6,
            "minimum_brightness": 40.0,
            "maximum_brightness": 210.0,
            "minimum_sharpness": 25.0,
            "store_embeddings": False,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            config = MonitorConfig.load(path)

        self.assertEqual(config.sample_interval_seconds, 5.0)
        self.assertEqual(config.detector_input_size, 320)
        self.assertFalse(config.store_embeddings)

    def test_builds_fixed_capture_region(self):
        region = region_from_args([10, 20, 800, 600])

        self.assertEqual(region.bbox, (10, 20, 810, 620))

    def test_rejects_invalid_fixed_capture_region(self):
        with self.assertRaises(ValueError):
            region_from_args([10, 20, 0, 600])


if __name__ == "__main__":
    unittest.main()
