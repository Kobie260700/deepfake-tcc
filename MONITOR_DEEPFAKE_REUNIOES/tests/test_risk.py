import unittest

from src.deepfake_detection.base import DeepfakeResult
from src.identity.profile import IdentityResult
from src.liveness.passive import LivenessResult
from src.risk.engine import RiskEngine


class RiskTests(unittest.TestCase):
    def test_accumulates_high_risk(self):
        engine = RiskEngine(65.0, 80.0, 20.0)
        identity = IdentityResult(30.0, False, 0.0)
        deepfake = DeepfakeResult(0.95, True, "teste")
        liveness = LivenessResult(0.0, 0.0, False, 0.0)

        assessments = [engine.assess(identity, deepfake, liveness, 1.0) for _ in range(3)]
        self.assertEqual(assessments[-1].level, "ALTO")
        self.assertEqual(assessments[-1].anomalies, 3)

    def test_supports_missing_detector(self):
        engine = RiskEngine(65.0, 80.0, 20.0)
        assessment = engine.assess(
            IdentityResult(2.0, True, 0.9),
            DeepfakeResult(None, False, "indisponivel"),
            LivenessResult(3.0, 0.01, True, 0.9),
            1.0,
        )
        self.assertEqual(assessment.level, "BAIXO")
        self.assertIsNone(assessment.manipulation_risk)

    def test_detector_cannot_hide_identity_mismatch(self):
        engine = RiskEngine(65.0, 80.0, 20.0)
        assessment = engine.assess(
            IdentityResult(30.0, False, 0.0),
            DeepfakeResult(0.01, True, "teste"),
            LivenessResult(4.0, 0.01, True, 1.0),
            1.0,
            active_challenge=True,
        )
        self.assertGreaterEqual(assessment.instant, 75.0)


if __name__ == "__main__":
    unittest.main()
