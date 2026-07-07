import unittest

from src.deepfake_detection.base import DeepfakeResult
from src.identity.profile import IdentityResult
from src.liveness.passive import LivenessResult
from src.risk.engine import RiskAssessment
from src.session_conclusion import build_final_conclusion, friendly_cycle_result


def assessment(level: str, compatible: bool = True, probability: float = 0.1):
    accumulated = {"BAIXO": 15.0, "MODERADO": 68.0, "ALTO": 85.0}[level]
    return {
        "identity": IdentityResult(5.0, compatible, 0.8),
        "deepfake": DeepfakeResult(
            probability,
            True,
            "teste",
            calibrated_probability=probability,
        ),
        "liveness": LivenessResult(2.0, 0.01, True, 0.8),
        "risk": RiskAssessment(
            accumulated, accumulated, level, 10.0, probability * 100, 20.0, 1.0, 0, None
        ),
    }


class SessionConclusionTests(unittest.TestCase):
    def test_low_risk_session_has_friendly_conclusion(self):
        result = build_final_conclusion([assessment("BAIXO")], "Pessoa")
        self.assertEqual(result["status"], "SEM_ALERTAS_RELEVANTES")
        self.assertEqual(result["cycles"], 1)

    def test_high_risk_session_requests_attention(self):
        result = build_final_conclusion([assessment("ALTO", False, 0.9)], "Pessoa")
        self.assertEqual(result["status"], "ATENCAO")
        self.assertIn("manipulacao facial", result["message"])
        self.assertIn("identidade", result["message"])
        self.assertEqual(friendly_cycle_result(result := assessment("ALTO")["risk"])[0][:7], "Atencao")

    def test_high_manipulation_is_reported_with_compatible_identity(self):
        result = build_final_conclusion([assessment("MODERADO", True, 0.86)], "Pessoa")
        self.assertEqual(result["status"], "ATENCAO")
        self.assertIn("manipulacao facial", result["message"])

    def test_counts_inconclusive_periods(self):
        result = build_final_conclusion(
            [assessment("BAIXO")],
            "Pessoa",
            total_periods=3,
        )
        self.assertEqual(result["completed_cycles"], 1)
        self.assertEqual(result["inconclusive_cycles"], 2)

    def test_reports_all_periods_when_none_are_complete(self):
        result = build_final_conclusion([], "Pessoa", total_periods=2)
        self.assertEqual(result["cycles"], 2)
        self.assertEqual(result["inconclusive_cycles"], 2)


if __name__ == "__main__":
    unittest.main()
