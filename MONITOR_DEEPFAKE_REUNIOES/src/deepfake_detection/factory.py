import json
from pathlib import Path

from src.deepfake_detection.base import EnsembleDetector, UnavailableDetector
from src.deepfake_detection.effort_detector import EffortSubprocessDetector
from src.deepfake_detection.onnx_detector import OnnxDeepfakeDetector


def load_detector(models_dir: Path):
    config_path = models_dir / "detector.json"
    if not config_path.exists():
        return UnavailableDetector(
            "Nenhum modelo configurado. O risco atual usa identidade e liveness."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = models_dir / config["file"]
    if not model_path.exists():
        return UnavailableDetector(f"Modelo ausente: {model_path.name}")

    try:
        primary = OnnxDeepfakeDetector(
            model_path=model_path,
            input_size=tuple(config.get("input_size", [224, 224])),
            model_name=config.get("name", model_path.stem),
            fake_class_index=int(config.get("fake_class_index", 1)),
            normalization=config.get("normalization", "imagenet"),
            resize_mode=config.get("resize_mode", "stretch"),
            output_type=config.get("output_type", "auto"),
        )
        effort_config_path = models_dir / "effort.json"
        if not effort_config_path.exists():
            return primary
        effort_config = json.loads(effort_config_path.read_text(encoding="utf-8"))
        project_root = models_dir.parent

        def resolve(value: str) -> Path:
            path = Path(value)
            return (path if path.is_absolute() else project_root / path).resolve()

        effort = EffortSubprocessDetector(
            python_path=resolve(effort_config["python"]),
            worker_path=resolve(effort_config.get("worker", "tools/effort_worker.py")),
            repository_path=resolve(effort_config["repository"]),
            checkpoint_path=resolve(effort_config["checkpoint"]),
            base_model_path=resolve(effort_config["base_model"]),
            timeout_seconds=float(effort_config.get("timeout_seconds", 180)),
            max_crops=int(effort_config.get("max_crops", 4)),
        )
        return EnsembleDetector([effort, primary])
    except Exception as error:
        return UnavailableDetector(f"Falha ao carregar {model_path.name}: {error}")
