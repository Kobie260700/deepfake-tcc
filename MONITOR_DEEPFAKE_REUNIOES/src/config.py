import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonitorConfig:
    sample_interval_seconds: float = 30.0
    analysis_window_samples: int = 8
    target_capture_fps: float = 12.0
    detector_input_size: int = 640
    minimum_face_size: int = 80
    minimum_detection_score: float = 0.5
    minimum_brightness: float = 35.0
    maximum_brightness: float = 220.0
    minimum_sharpness: float = 30.0
    monitor_minimum_sharpness: float = 4.0
    identity_distance_threshold: float = 20.21
    identity_warning_ratio: float = 0.85
    effort_delta_warning: float = 0.015
    effort_delta_alert: float = 0.030
    minimum_temporal_variation: float = 1.0
    minimum_landmark_movement: float = 0.0005
    maximum_landmark_movement: float = 0.08
    risk_alert_threshold: float = 65.0
    risk_high_threshold: float = 80.0
    store_embeddings: bool = True
    show_preview: bool = False
    show_status_overlay: bool = True

    @classmethod
    def load(cls, path: Path) -> "MonitorConfig":
        with path.open("r", encoding="utf-8") as file:
            values = json.load(file)
        return cls(**values)
