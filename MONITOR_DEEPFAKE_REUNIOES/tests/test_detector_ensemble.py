import unittest

from src.deepfake_detection.base import DeepfakeResult, EnsembleDetector, calibrate_result


class StubDetector:
    def __init__(self, result):
        self.result = result

    def predict(self, face_crops, aligned_face_crops=None):
        return self.result


class DetectorEnsembleTests(unittest.TestCase):
    def test_calibrates_effort_against_reference_baseline(self):
        raw = DeepfakeResult(
            0.83,
            True,
            "Effort + DFDC",
            effort_probability=0.83,
            component_probabilities={"effort": 0.83, "dfdc": 0.02},
        )
        real_like = calibrate_result(raw, effort_baseline=0.82)
        fake_like = calibrate_result(raw, effort_baseline=0.79)
        self.assertEqual(real_like.calibrated_probability, 0.02)
        self.assertEqual(fake_like.calibrated_probability, 1.0)

    def test_reports_ready_fallback_without_samples(self):
        detector = EnsembleDetector([
            StubDetector(DeepfakeResult(None, False, "Effort", "peso ausente")),
            StubDetector(DeepfakeResult(None, True, "DFDC", "Sem recortes validos.")),
        ])
        result = detector.predict([])
        self.assertTrue(result.available)
        self.assertEqual(result.model_name, "DFDC")
        self.assertIn("peso ausente", result.detail)

    def test_uses_highest_available_probability(self):
        detector = EnsembleDetector([
            StubDetector(DeepfakeResult(0.82, True, "Effort")),
            StubDetector(DeepfakeResult(0.03, True, "DFDC")),
        ])
        result = detector.predict([object()])
        self.assertEqual(result.probability, 0.82)
        self.assertIn("Effort", result.model_name)
        self.assertIn("DFDC", result.model_name)

    def test_falls_back_when_optional_detector_is_unavailable(self):
        detector = EnsembleDetector([
            StubDetector(DeepfakeResult(None, False, "Effort", "peso ausente")),
            StubDetector(DeepfakeResult(0.07, True, "DFDC")),
        ])
        result = detector.predict([object()])
        self.assertEqual(result.probability, 0.07)
        self.assertEqual(result.model_name, "DFDC")
        self.assertIn("peso ausente", result.detail)


if __name__ == "__main__":
    unittest.main()
