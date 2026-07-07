from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DeepfakeResult:
    probability: float | None
    available: bool
    model_name: str
    detail: str = ""
    effort_probability: float | None = None
    effort_baseline: float | None = None
    effort_delta: float | None = None
    calibrated_probability: float | None = None
    component_probabilities: dict[str, float] = field(default_factory=dict)


class DeepfakeDetector(Protocol):
    def predict(
        self,
        face_crops: list[np.ndarray],
        aligned_face_crops: list[np.ndarray] | None = None,
    ) -> DeepfakeResult:
        ...


class UnavailableDetector:
    def __init__(self, detail: str = "Detector pre-treinado ainda nao configurado."):
        self.detail = detail

    def predict(
        self,
        face_crops: list[np.ndarray],
        aligned_face_crops: list[np.ndarray] | None = None,
    ) -> DeepfakeResult:
        return DeepfakeResult(None, False, "indisponivel", self.detail)


class EnsembleDetector:
    """Combines detectors conservatively while allowing optional models to fail."""

    def __init__(self, detectors: list[DeepfakeDetector]):
        self.detectors = detectors

    def predict(
        self,
        face_crops: list[np.ndarray],
        aligned_face_crops: list[np.ndarray] | None = None,
    ) -> DeepfakeResult:
        results = [
            detector.predict(face_crops, aligned_face_crops)
            for detector in self.detectors
        ]
        if not face_crops:
            ready = [result for result in results if result.available]
            details = "; ".join(result.detail for result in results if result.detail)
            if ready:
                return DeepfakeResult(
                    None,
                    True,
                    " + ".join(result.model_name for result in ready),
                    details,
                )
            return DeepfakeResult(None, False, "ensemble indisponivel", details)
        available = [
            result
            for result in results
            if result.available and result.probability is not None
        ]
        if not available:
            detail = "; ".join(result.detail for result in results if result.detail)
            return DeepfakeResult(None, False, "ensemble indisponivel", detail)

        selected = max(available, key=lambda result: float(result.probability))
        names = " + ".join(result.model_name for result in available)
        details = [
            f"{result.model_name}={float(result.probability):.4f}"
            for result in available
        ]
        unavailable = [result.detail for result in results if not result.available and result.detail]
        components = {}
        effort_probability = None
        for result in available:
            components.update(result.component_probabilities)
            if result.effort_probability is not None:
                effort_probability = result.effort_probability
        return DeepfakeResult(
            selected.probability,
            True,
            names,
            "; ".join(details + unavailable),
            effort_probability=effort_probability,
            component_probabilities=components,
        )


def calibrate_result(
    result: DeepfakeResult,
    effort_baseline: float | None,
    delta_warning: float = 0.015,
    delta_alert: float = 0.030,
) -> DeepfakeResult:
    """Convert Effort's domain-dependent softmax into a reference-relative score."""
    fallback_probability = result.component_probabilities.get("dfdc")
    calibrated = fallback_probability
    delta = None
    if result.effort_probability is not None and effort_baseline is not None:
        delta = result.effort_probability - effort_baseline
        denominator = max(delta_alert - delta_warning, 1e-6)
        effort_score = float(np.clip((delta - delta_warning) / denominator, 0.0, 1.0))
        calibrated = max(effort_score, fallback_probability or 0.0)
    return replace(
        result,
        effort_baseline=effort_baseline,
        effort_delta=delta,
        calibrated_probability=calibrated,
    )
