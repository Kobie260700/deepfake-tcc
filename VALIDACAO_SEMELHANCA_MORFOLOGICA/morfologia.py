from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class Sample:
    source: str
    frame_index: int
    detection_score: float
    sharpness: float
    brightness: float
    face_size: int
    landmarks: list[list[float]]


@dataclass
class SetAnalysis:
    name: str
    files: list[str]
    samples: int
    rejected_frames: int
    median_shape: list[list[float]]
    internal_shape_rmse: float
    ratios: dict[str, dict[str, float]]


def normalize_shape(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)[:, :2]
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.sum(centered**2)))
    if scale <= 1e-12:
        raise ValueError("Landmarks degenerados.")
    return centered / scale


def align_shape(shape: np.ndarray, target: np.ndarray) -> np.ndarray:
    shape = normalize_shape(shape)
    target = normalize_shape(target)
    u, _, vt = np.linalg.svd(shape.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return shape @ rotation


def generalized_mean_shape(shapes: Iterable[np.ndarray], iterations: int = 8) -> np.ndarray:
    normalized = [normalize_shape(shape) for shape in shapes]
    if not normalized:
        raise ValueError("Nenhuma forma facial valida.")
    target = normalized[0]
    for _ in range(iterations):
        aligned = np.stack([align_shape(shape, target) for shape in normalized])
        target = normalize_shape(np.mean(aligned, axis=0))
    return target


def shape_rmse(shape_a: np.ndarray, shape_b: np.ndarray) -> float:
    a = align_shape(shape_a, shape_b)
    b = normalize_shape(shape_b)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def facial_ratios(points: np.ndarray) -> dict[str, float]:
    """Ratios for the standard InsightFace 68-point landmark ordering."""
    points = np.asarray(points, dtype=np.float64)
    width = _distance(points, 0, 16)
    height = _distance(points, 8, 27)
    if min(width, height) <= 1e-9:
        return {}
    eye_left = np.mean(points[36:42, :2], axis=0)
    eye_right = np.mean(points[42:48, :2], axis=0)
    return {
        "largura_altura_rosto": width / height,
        "distancia_olhos_largura_rosto": float(np.linalg.norm(eye_left - eye_right)) / width,
        "largura_nariz_largura_rosto": _distance(points, 31, 35) / width,
        "largura_boca_largura_rosto": _distance(points, 48, 54) / width,
        "altura_terco_inferior_altura_rosto": _distance(points, 33, 8) / height,
        "largura_mandibula_largura_rosto": _distance(points, 4, 12) / width,
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "media": float(np.mean(array)),
        "desvio_padrao": float(np.std(array)),
        "mediana": float(np.median(array)),
    }


class MorphologyAnalyzer:
    def __init__(self, detector_size: int = 640):
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "landmark_3d_68"],
        )
        self.app.prepare(ctx_id=0, det_size=(detector_size, detector_size))

    def collect(
        self,
        files: list[Path],
        max_samples: int = 40,
        video_interval_seconds: float = 0.5,
    ) -> tuple[list[Sample], int]:
        samples: list[Sample] = []
        rejected = 0
        for path in files:
            if len(samples) >= max_samples:
                break
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(path))
                sample = self._sample(frame, path, 0) if frame is not None else None
                if sample is None:
                    rejected += 1
                else:
                    samples.append(sample)
            elif suffix in VIDEO_EXTENSIONS:
                capture = cv2.VideoCapture(str(path))
                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                step = max(1, int(fps * video_interval_seconds))
                index = 0
                while len(samples) < max_samples:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % step == 0:
                        sample = self._sample(frame, path, index)
                        if sample is None:
                            rejected += 1
                        else:
                            samples.append(sample)
                    index += 1
                capture.release()
            else:
                rejected += 1
        return samples, rejected

    def _sample(self, frame: np.ndarray, source: Path, frame_index: int) -> Sample | None:
        faces = list(self.app.get(frame))
        if len(faces) != 1:
            return None
        face = faces[0]
        landmarks = getattr(face, "landmark_3d_68", None)
        if landmarks is None:
            return None
        x1, y1, x2, y2 = np.round(face.bbox).astype(int)
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        face_size = int(min(x2 - x1, y2 - y1))
        if float(face.det_score) < 0.55 or face_size < 90 or sharpness < 20 or not 30 <= brightness <= 225:
            return None
        return Sample(
            source=str(source),
            frame_index=frame_index,
            detection_score=float(face.det_score),
            sharpness=sharpness,
            brightness=brightness,
            face_size=face_size,
            landmarks=np.asarray(landmarks, dtype=float).tolist(),
        )


def analyze_set(name: str, files: list[Path], samples: list[Sample], rejected: int) -> SetAnalysis:
    if not samples:
        raise ValueError(f"{name}: nenhuma amostra facial valida foi encontrada.")
    shapes = [np.asarray(sample.landmarks) for sample in samples]
    mean_shape = generalized_mean_shape(shapes)
    aligned = np.stack([align_shape(shape, mean_shape) for shape in shapes])
    median_shape = normalize_shape(np.median(aligned, axis=0))
    internal = [shape_rmse(shape, median_shape) for shape in shapes]
    ratio_rows = [facial_ratios(shape) for shape in shapes]
    ratio_names = ratio_rows[0].keys()
    ratios = {key: _summary([row[key] for row in ratio_rows]) for key in ratio_names}
    return SetAnalysis(
        name=name,
        files=[str(path) for path in files],
        samples=len(samples),
        rejected_frames=rejected,
        median_shape=median_shape.tolist(),
        internal_shape_rmse=float(np.mean(internal)),
        ratios=ratios,
    )


def _generation_prompt(profile: SetAnalysis) -> str:
    values = {name: data["media"] for name, data in profile.ratios.items()}
    return f"""Crie um retrato frontal, fotorealista e neutro de uma pessoa adulta ficticia.
A identidade gerada deve ser original e biometricamente distinta da pessoa de
referencia. Utilize apenas as proporcoes morfologicas abaixo como orientacao de
estrutura facial, sem reproduzir marcas, textura de pele ou detalhes exclusivos:

- razao largura/altura do rosto: {values['largura_altura_rosto']:.4f}
- distancia entre os olhos/largura do rosto: {values['distancia_olhos_largura_rosto']:.4f}
- largura do nariz/largura do rosto: {values['largura_nariz_largura_rosto']:.4f}
- largura da boca/largura do rosto: {values['largura_boca_largura_rosto']:.4f}
- terco facial inferior/altura do rosto: {values['altura_terco_inferior_altura_rosto']:.4f}
- largura da mandibula/largura do rosto: {values['largura_mandibula_largura_rosto']:.4f}

Mantenha pose frontal, expressao neutra, iluminacao uniforme e fundo simples.
Nao replique a identidade da imagem original; gere uma pessoa inexistente com
estrutura facial aproximadamente semelhante para uso academico controlado.
"""


def save_profile_reports(
    output: Path,
    profile: SetAnalysis,
    samples: list[Sample],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "metodologia": {
            "modelo": "InsightFace buffalo_l",
            "landmarks": "landmark_3d_68 projetados em 2D e normalizados por Procrustes",
            "natureza": "extracao descritiva de proporcoes morfologicas",
            "observacao": "embeddings biometricos nao sao calculados nem exportados",
        },
        "perfil_morfologico": asdict(profile),
    }
    (output / "relatorio_completo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "proporcoes_faciais.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["metrica", "media", "mediana", "desvio_padrao"])
        for name, row in profile.ratios.items():
            writer.writerow([name, row["media"], row["mediana"], row["desvio_padrao"]])
    with (output / "amostras.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["arquivo", "frame", "deteccao", "nitidez", "brilho", "tamanho_rosto"])
        for sample in samples:
            writer.writerow([
                sample.source, sample.frame_index, sample.detection_score,
                sample.sharpness, sample.brightness, sample.face_size,
            ])
    ratio_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {row['media']:.4f} | "
        f"{row['mediana']:.4f} | {row['desvio_padrao']:.4f} |"
        for name, row in profile.ratios.items()
    )
    reliability = (
        "baixa: apenas uma amostra valida"
        if profile.samples == 1
        else "moderada" if profile.samples < 10 else "mais estavel para este conjunto"
    )
    markdown = f"""# Ficha morfologica facial

## Amostras

- Amostras validas: {profile.samples}.
- Quadros rejeitados: {profile.rejected_frames}.
- Confiabilidade descritiva: {reliability}.

- Variabilidade interna da forma normalizada: {profile.internal_shape_rmse:.6f}.

## Proporcoes

| Metrica | Media | Mediana | Desvio padrao |
|---|---:|---:|---:|
{ratio_lines}

## Interpretacao metodologica

As medidas sao razoes adimensionais extraidas de 68 pontos faciais. Elas servem
como especificacao de estrutura e nao como identificadores biometricos. Nenhum
embedding facial e exportado. Uma unica imagem permite a extracao, mas imagens
e videos adicionais reduzem a influencia de pose, expressao e erro de deteccao.
"""
    (output / "FICHA_MORFOLOGICA.md").write_text(markdown, encoding="utf-8")
    (output / "PROMPT_GERACAO.txt").write_text(_generation_prompt(profile), encoding="utf-8")
