import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.identity.profile import IdentityProfile, IdentityVerifier, safe_profile_name


class IdentityTests(unittest.TestCase):
    def test_profile_round_trip_and_verification(self):
        embeddings = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
        profile = IdentityProfile.create("Pessoa Teste", embeddings, source_count=2)

        with tempfile.TemporaryDirectory() as directory:
            profile_dir = profile.save(Path(directory))
            loaded = IdentityProfile.load(profile_dir)

        result = IdentityVerifier(loaded, threshold=3.0).verify([np.zeros(4, dtype=np.float32)])
        self.assertEqual(loaded.name, "Pessoa Teste")
        self.assertTrue(result.compatible)
        self.assertEqual(safe_profile_name("Pessoa Teste"), "pessoa_teste")
        self.assertEqual(loaded.registration_method, "images")

    def test_rejects_distant_identity(self):
        profile = IdentityProfile.create("A", [np.zeros(3, dtype=np.float32)], 1)
        result = IdentityVerifier(profile, threshold=2.0).verify(
            [np.full(3, 10.0, dtype=np.float32)]
        )
        self.assertFalse(result.compatible)

    def test_preserves_submitted_video_metadata(self):
        profile = IdentityProfile.create(
            "Candidato",
            [np.zeros(3, dtype=np.float32)],
            source_count=10,
            registration_method="submitted_video",
            source_deepfake_probability=0.12,
            source_effort_probability=0.78,
        )

        with tempfile.TemporaryDirectory() as directory:
            loaded = IdentityProfile.load(profile.save(Path(directory)))

        self.assertEqual(loaded.registration_method, "submitted_video")
        self.assertEqual(loaded.source_deepfake_probability, 0.12)
        self.assertEqual(loaded.source_effort_probability, 0.78)


if __name__ == "__main__":
    unittest.main()
