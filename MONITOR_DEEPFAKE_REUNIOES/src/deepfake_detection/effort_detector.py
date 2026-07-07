import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.deepfake_detection.base import DeepfakeResult


class EffortSubprocessDetector:
    """Runs the heavy Effort model in an isolated persistent Python process."""

    def __init__(
        self,
        python_path: Path,
        worker_path: Path,
        repository_path: Path,
        checkpoint_path: Path,
        base_model_path: Path,
        timeout_seconds: float = 180.0,
        max_crops: int = 4,
    ):
        self.python_path = python_path
        self.worker_path = worker_path
        self.repository_path = repository_path
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path
        self.timeout_seconds = timeout_seconds
        self.max_crops = max(1, max_crops)
        self.process: subprocess.Popen[str] | None = None
        self.unavailable_reason = self._validate_files()
        if self.unavailable_reason is None:
            self._start()

    def _validate_files(self) -> str | None:
        required = {
            "Python isolado": self.python_path,
            "executor Effort": self.worker_path,
            "repositorio Effort": self.repository_path,
            "checkpoint Effort": self.checkpoint_path,
            "CLIP-L/14": self.base_model_path,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        return "Arquivos ausentes - " + " | ".join(missing) if missing else None

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        command = [
            str(self.python_path),
            str(self.worker_path),
            "--repository", str(self.repository_path),
            "--checkpoint", str(self.checkpoint_path),
            "--base-model", str(self.base_model_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def predict(
        self,
        face_crops: list[np.ndarray],
        aligned_face_crops: list[np.ndarray] | None = None,
    ) -> DeepfakeResult:
        if aligned_face_crops:
            face_crops = aligned_face_crops
        if not face_crops:
            if self.unavailable_reason:
                return DeepfakeResult(None, False, "Effort CLIP-L/14", self.unavailable_reason)
            return DeepfakeResult(None, True, "Effort CLIP-L/14", "Pronto para analise.")
        if self.unavailable_reason:
            return DeepfakeResult(None, False, "Effort CLIP-L/14", self.unavailable_reason)

        try:
            self._start()
            assert self.process is not None
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            with tempfile.TemporaryDirectory(prefix="effort_faces_") as directory:
                if len(face_crops) > self.max_crops:
                    indices = np.linspace(
                        0,
                        len(face_crops) - 1,
                        self.max_crops,
                        dtype=int,
                    )
                    face_crops = [face_crops[index] for index in indices]
                paths = []
                for index, crop in enumerate(face_crops):
                    path = Path(directory) / f"face_{index:02d}.jpg"
                    if cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        paths.append(str(path))
                self.process.stdin.write(json.dumps({"paths": paths}) + "\n")
                self.process.stdin.flush()
                response = self.process.stdout.readline()
            if not response:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(stderr.strip() or "O processo Effort foi encerrado.")
            payload = json.loads(response)
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            probabilities = [float(value) for value in payload["probabilities"]]
            probability = float(np.median(probabilities))
            return DeepfakeResult(
                probability,
                True,
                "Effort CLIP-L/14 (ICML 2025)",
                f"{len(probabilities)} quadros; max={max(probabilities):.4f}",
                effort_probability=probability,
                component_probabilities={"effort": probability},
            )
        except Exception as error:
            self.close()
            return DeepfakeResult(None, False, "Effort CLIP-L/14", f"Falha no Effort: {error}")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None

    def __del__(self):
        self.close()
