import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.face_tracking.analyzer import FaceQuality
from src.storage.session import SessionRecorder


class SessionRecorderTests(unittest.TestCase):
    def test_records_metrics_and_embedding_without_saving_frame(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        quality = FaceQuality(True, 0.98, 100, 110.0, 80.0, ())
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory), store_embeddings=True)
            recorder.record(frame, quality, embedding)
            recorder.record_period(1, False, 1, 20, "Amostras insuficientes")

            with recorder.metrics_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            entries = [
                json.loads(line)
                for line in recorder.embeddings_path.read_text(encoding="utf-8").splitlines()
            ]
            image_files = list(recorder.session_dir.glob("*.png")) + list(
                recorder.session_dir.glob("*.jpg")
            )
            periods = [
                json.loads(line)
                for line in recorder.periods_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["largura_frame"], "200")
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0]["embedding"]), 3)
        self.assertEqual(image_files, [])
        self.assertEqual(periods[0]["periodo"], 1)
        self.assertFalse(periods[0]["analise_completa"])


if __name__ == "__main__":
    unittest.main()
