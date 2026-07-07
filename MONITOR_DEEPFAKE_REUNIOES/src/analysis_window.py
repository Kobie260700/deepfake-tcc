from collections import deque
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowSnapshot:
    embeddings: list[np.ndarray]
    landmarks: list[np.ndarray]
    gray_crops: list[np.ndarray]
    color_crops: list[np.ndarray]
    aligned_color_crops: list[np.ndarray]
    quality_ratio: float


class AnalysisWindow:
    def __init__(self, maximum_samples: int):
        self.samples = deque(maxlen=maximum_samples)
        self.total_frames = 0
        self.valid_frames = 0
        self.rejection_reasons = Counter()

    def add(
        self,
        embedding,
        landmarks,
        gray_crop,
        color_crop,
        aligned_color_crop,
        quality_adequate: bool,
    ) -> None:
        self.total_frames += 1
        if (
            not quality_adequate
            or gray_crop is None
            or color_crop is None
            or aligned_color_crop is None
        ):
            if quality_adequate:
                self.rejection_reasons["qualidade_insuficiente"] += 1
            return
        self.valid_frames += 1
        self.samples.append((embedding, landmarks, gray_crop, color_crop, aligned_color_crop))

    def add_invalid(self, reason: str = "rosto_nao_detectado") -> None:
        self.total_frames += 1
        self.rejection_reasons[reason] += 1

    def add_quality_reasons(self, reasons: tuple[str, ...]) -> None:
        for reason in reasons:
            self.rejection_reasons[reason] += 1

    def main_rejection_reason(self) -> str | None:
        if not self.rejection_reasons:
            return None
        return self.rejection_reasons.most_common(1)[0][0]

    def ready(self, minimum_samples: int = 3) -> bool:
        return len(self.samples) >= minimum_samples

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def snapshot(self) -> WindowSnapshot:
        if not self.samples:
            raise ValueError("A janela esta vazia.")
        embeddings, landmarks, gray_crops, color_crops, aligned_color_crops = zip(*self.samples)
        quality_ratio = self.valid_frames / max(self.total_frames, 1)
        return WindowSnapshot(
            list(embeddings),
            list(landmarks),
            list(gray_crops),
            list(color_crops),
            list(aligned_color_crops),
            quality_ratio,
        )

    def reset(self) -> None:
        self.samples.clear()
        self.total_frames = 0
        self.valid_frames = 0
        self.rejection_reasons.clear()
