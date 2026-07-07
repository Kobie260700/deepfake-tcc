import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.face_tracking.analyzer import FaceQuality
from src.deepfake_detection.base import DeepfakeResult
from src.identity.profile import IdentityResult
from src.liveness.passive import LivenessResult
from src.risk.engine import RiskAssessment


class SessionRecorder:
    def __init__(self, output_root: Path, store_embeddings: bool):
        session_name = datetime.now().strftime("sessao_%Y%m%d_%H%M%S")
        self.session_dir = output_root / session_name
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.session_dir / "metricas.csv"
        self.embeddings_path = self.session_dir / "embeddings.jsonl"
        self.assessments_path = self.session_dir / "avaliacoes.csv"
        self.periods_path = self.session_dir / "periodos.jsonl"
        self.store_embeddings = store_embeddings
        self._create_metrics_file()
        self._create_assessments_file()

    def write_session_info(self, information: dict) -> None:
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **information,
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_conclusion(self, conclusion: dict) -> None:
        (self.session_dir / "conclusao.json").write_text(
            json.dumps(conclusion, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _create_metrics_file(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "data_hora",
                "face_detectada",
                "qualidade_adequada",
                "motivos_qualidade",
                "confianca_deteccao",
                "tamanho_face",
                "brilho",
                "nitidez",
                "largura_frame",
                "altura_frame",
            ])

    def _create_assessments_file(self) -> None:
        with self.assessments_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "data_hora", "distancia_identidade", "identidade_compativel",
                "probabilidade_deepfake", "detector_disponivel", "modelo_detector",
                "probabilidade_calibrada", "probabilidade_effort",
                "baseline_effort", "delta_effort",
                "variacao_temporal", "movimento_landmarks", "liveness_valido",
                "risco_instantaneo", "risco_acumulado", "nivel_risco",
                "confianca_analise", "anomalias_consecutivas",
                "challenge_ativo_resultado",
            ])

    def record(self, frame: np.ndarray, quality: FaceQuality, embedding: np.ndarray) -> None:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        height, width = frame.shape[:2]

        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                timestamp,
                True,
                quality.adequate,
                "|".join(quality.reasons),
                round(quality.detection_score, 4),
                quality.face_size,
                round(quality.brightness, 3),
                round(quality.sharpness, 3),
                width,
                height,
            ])

        if self.store_embeddings:
            entry = {
                "data_hora": timestamp,
                "embedding": embedding.astype(float).tolist(),
            }
            with self.embeddings_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def record_assessment(
        self,
        identity: IdentityResult,
        deepfake: DeepfakeResult,
        liveness: LivenessResult,
        risk: RiskAssessment,
    ) -> None:
        with self.assessments_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                round(identity.distance, 4), identity.compatible,
                "" if deepfake.probability is None else round(deepfake.probability, 6),
                deepfake.available, deepfake.model_name,
                "" if deepfake.calibrated_probability is None else round(deepfake.calibrated_probability, 6),
                "" if deepfake.effort_probability is None else round(deepfake.effort_probability, 6),
                "" if deepfake.effort_baseline is None else round(deepfake.effort_baseline, 6),
                "" if deepfake.effort_delta is None else round(deepfake.effort_delta, 6),
                round(liveness.temporal_variation, 4),
                round(liveness.landmark_movement, 6), liveness.valid,
                round(risk.instant, 3), round(risk.accumulated, 3), risk.level,
                round(risk.confidence, 4), risk.anomalies,
                "" if risk.active_challenge is None else risk.active_challenge,
            ])

    def record_period(
        self,
        period_number: int,
        completed: bool,
        valid_samples: int,
        total_frames: int,
        message: str,
    ) -> None:
        entry = {
            "data_hora": datetime.now().isoformat(timespec="milliseconds"),
            "periodo": period_number,
            "analise_completa": completed,
            "amostras_validas": valid_samples,
            "quadros_observados": total_frames,
            "mensagem": message,
        }
        with self.periods_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
