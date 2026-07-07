import unittest

import numpy as np

from src.identity.registration import (
    _filter_consistent_embeddings,
    _validate_submitted_video_duration,
    RegistrationError,
)


class VideoRegistrationTests(unittest.TestCase):
    def test_filters_embedding_outlier(self):
        embeddings = [
            np.zeros(4, dtype=np.float32),
            np.full(4, 0.1, dtype=np.float32),
            np.full(4, 100.0, dtype=np.float32),
        ]
        filtered = _filter_consistent_embeddings(embeddings)
        self.assertEqual(len(filtered), 2)

    def test_accepts_expected_submitted_video_duration(self):
        _validate_submitted_video_duration(12.0)

    def test_rejects_excessive_submitted_video_duration(self):
        with self.assertRaises(RegistrationError):
            _validate_submitted_video_duration(45.0)


if __name__ == "__main__":
    unittest.main()
