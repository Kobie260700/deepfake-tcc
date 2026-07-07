import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def safe_profile_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    if not normalized:
        raise ValueError("O nome do perfil nao pode ser vazio.")
    return normalized.lower()


@dataclass(frozen=True)
class IdentityProfile:
    name: str
    embeddings: np.ndarray
    created_at: str
    source_count: int
    registration_method: str = "images"
    source_deepfake_probability: float | None = None
    source_effort_probability: float | None = None

    @property
    def reference_embedding(self) -> np.ndarray:
        return np.median(self.embeddings, axis=0).astype(np.float32)

    def save(self, profiles_dir: Path) -> Path:
        profile_dir = profiles_dir / safe_profile_name(self.name)
        profile_dir.mkdir(parents=True, exist_ok=True)
        np.save(profile_dir / "embeddings.npy", self.embeddings.astype(np.float32))
        metadata = {
            "name": self.name,
            "created_at": self.created_at,
            "source_count": self.source_count,
            "embedding_count": int(len(self.embeddings)),
            "embedding_dimension": int(self.embeddings.shape[1]),
            "model": "InsightFace buffalo_l",
            "registration_method": self.registration_method,
            "source_deepfake_probability": self.source_deepfake_probability,
            "source_effort_probability": self.source_effort_probability,
        }
        (profile_dir / "profile.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return profile_dir

    @classmethod
    def load(cls, profile_dir: Path) -> "IdentityProfile":
        metadata = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
        embeddings = np.load(profile_dir / "embeddings.npy").astype(np.float32)
        if embeddings.ndim != 2 or not len(embeddings):
            raise ValueError("O perfil nao possui embeddings validos.")
        return cls(
            name=metadata["name"],
            embeddings=embeddings,
            created_at=metadata["created_at"],
            source_count=int(metadata["source_count"]),
            registration_method=metadata.get("registration_method", "images"),
            source_deepfake_probability=metadata.get("source_deepfake_probability"),
            source_effort_probability=metadata.get("source_effort_probability"),
        )

    @classmethod
    def create(
        cls,
        name: str,
        embeddings: list[np.ndarray],
        source_count: int,
        registration_method: str = "images",
        source_deepfake_probability: float | None = None,
        source_effort_probability: float | None = None,
    ) -> "IdentityProfile":
        if not embeddings:
            raise ValueError("Nenhum embedding valido foi fornecido.")
        matrix = np.stack(embeddings).astype(np.float32)
        return cls(
            name,
            matrix,
            datetime.now().isoformat(timespec="seconds"),
            source_count,
            registration_method,
            source_deepfake_probability,
            source_effort_probability,
        )


@dataclass(frozen=True)
class IdentityResult:
    distance: float
    compatible: bool
    confidence: float


class IdentityVerifier:
    def __init__(self, profile: IdentityProfile, threshold: float):
        self.profile = profile
        self.threshold = threshold
        self.reference = profile.reference_embedding

    def verify(self, embeddings: list[np.ndarray]) -> IdentityResult:
        if not embeddings:
            raise ValueError("A janela nao possui embeddings.")
        representative = np.median(np.stack(embeddings), axis=0)
        distance = float(np.linalg.norm(self.reference - representative))
        confidence = float(np.clip(1.0 - distance / max(self.threshold, 1e-6), 0.0, 1.0))
        return IdentityResult(distance, distance <= self.threshold, confidence)
