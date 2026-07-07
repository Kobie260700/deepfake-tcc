from dataclasses import dataclass

import numpy as np

from src.deepfake_detection.base import DeepfakeResult
from src.identity.profile import IdentityResult
from src.liveness.passive import LivenessResult


@dataclass(frozen=True)
class RiskAssessment:
    instant: float
    accumulated: float
    level: str
    identity_risk: float
    manipulation_risk: float | None
    liveness_risk: float
    confidence: float
    anomalies: int
    active_challenge: bool | None


class RiskEngine:
    def __init__(
        self,
        alert_threshold: float,
        high_threshold: float,
        identity_threshold: float,
        identity_warning_ratio: float = 0.85,
    ):
        self.alert_threshold = alert_threshold
        self.high_threshold = high_threshold
        self.identity_threshold = identity_threshold
        self.identity_warning_ratio = identity_warning_ratio
        self.history = []
        self.consecutive_anomalies = 0

    def assess(
        self,
        identity: IdentityResult,
        deepfake: DeepfakeResult,
        liveness: LivenessResult,
        quality_ratio: float,
        active_challenge: bool | None = None,
    ) -> RiskAssessment:
        identity_risk = float(
            np.clip(identity.distance / max(self.identity_threshold, 1e-6), 0.0, 1.0) * 100
        )
        liveness_risk = float((1.0 - liveness.confidence) * 100)
        if deepfake.available and deepfake.probability is not None:
            risk_probability = (
                deepfake.calibrated_probability
                if deepfake.calibrated_probability is not None
                else deepfake.probability
            )
            manipulation_risk = float(risk_probability * 100)
            instant = 0.45 * identity_risk + 0.40 * manipulation_risk + 0.15 * liveness_risk
        else:
            manipulation_risk = None
            instant = 0.70 * identity_risk + 0.30 * liveness_risk

        if active_challenge is True:
            instant = max(0.0, instant - 10.0)
        elif active_challenge is False:
            instant = min(100.0, instant + 20.0)

        confidence = float(np.clip(quality_ratio, 0.0, 1.0))
        instant *= 0.65 + 0.35 * confidence

        # Evidencias fortes nao podem ser diluidas pela media ponderada.
        if not identity.compatible:
            instant = max(instant, 75.0)
        elif identity.distance >= self.identity_threshold * self.identity_warning_ratio:
            instant = max(instant, 50.0)
        if manipulation_risk is not None and manipulation_risk >= 80.0:
            instant = max(instant, 75.0)
        self.history.append(float(instant))
        self.history = self.history[-12:]
        accumulated = float(np.average(self.history, weights=np.arange(1, len(self.history) + 1)))

        if instant >= self.alert_threshold:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1)

        if accumulated >= self.high_threshold or self.consecutive_anomalies >= 3:
            level = "ALTO"
        elif accumulated >= self.alert_threshold:
            level = "MODERADO"
        else:
            level = "BAIXO"

        return RiskAssessment(
            instant=float(instant),
            accumulated=accumulated,
            level=level,
            identity_risk=identity_risk,
            manipulation_risk=manipulation_risk,
            liveness_risk=liveness_risk,
            confidence=confidence,
            anomalies=self.consecutive_anomalies,
            active_challenge=active_challenge,
        )
