from pathlib import Path

import cv2
import numpy as np

from src.face_tracking.analyzer import FaceMonitorAnalyzer
from src.deepfake_detection.base import DeepfakeDetector
from src.identity.profile import IdentityProfile
from src.liveness.active import ActiveChallenge
from src.liveness.passive import analyze_liveness, normalized_face_crop, normalized_landmarks


class RegistrationError(RuntimeError):
    pass


def _filter_consistent_embeddings(embeddings: list[np.ndarray]) -> list[np.ndarray]:
    if len(embeddings) <= 1:
        return embeddings

    reference = np.median(np.stack(embeddings), axis=0)
    distances = [float(np.linalg.norm(reference - embedding)) for embedding in embeddings]
    median_distance = float(np.median(distances))
    limit = max(median_distance * 2.5, 8.0)
    accepted = [
        embedding
        for embedding, distance in zip(embeddings, distances)
        if distance <= limit
    ]
    if not accepted:
        raise RegistrationError("As amostras nao possuem identidade consistente.")
    return accepted


def create_profile_from_images(
    name: str,
    image_paths: list[Path],
    analyzer: FaceMonitorAnalyzer,
) -> IdentityProfile:
    embeddings = []
    errors = []

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            errors.append(f"{path.name}: arquivo invalido")
            continue
        faces = analyzer.app.get(frame)
        if len(faces) != 1:
            errors.append(f"{path.name}: esperado 1 rosto, encontrados {len(faces)}")
            continue
        quality = analyzer.evaluate_quality(frame, faces[0])
        if not quality.adequate:
            errors.append(f"{path.name}: {', '.join(quality.reasons)}")
            continue
        embeddings.append(analyzer.embedding(faces[0]))

    if not embeddings:
        details = "; ".join(errors) or "nenhuma imagem processada"
        raise RegistrationError(f"Nao foi possivel criar o perfil: {details}")

    embeddings = _filter_consistent_embeddings(embeddings)

    return IdentityProfile.create(name, embeddings, len(image_paths))


class BlinkDetector:
    def __init__(self):
        self.open_samples: list[float] = []
        self.reference_ear: float | None = None
        self.was_closed = False
        self.completed = False

    def update(self, face) -> bool:
        points = getattr(face, "landmark_3d_68", None)
        if points is None:
            return False
        points = np.asarray(points, dtype=np.float32)[:, :2]
        ear = (
            ActiveChallenge._ear(points[36:42])
            + ActiveChallenge._ear(points[42:48])
        ) / 2
        if self.reference_ear is None:
            self.open_samples.append(ear)
            if len(self.open_samples) >= 8:
                self.reference_ear = float(np.median(self.open_samples))
            return False

        closed = ear < self.reference_ear * 0.72
        if closed:
            self.was_closed = True
        elif self.was_closed:
            self.completed = True
        return self.completed


def _validate_submitted_video_duration(duration: float) -> None:
    if duration < 8 or duration > 30:
        raise RegistrationError(
            f"O video possui {duration:.1f}s. Solicite um arquivo de aproximadamente 10 a 15 segundos."
        )


def create_profile_from_video_file(
    name: str,
    video_path: Path,
    analyzer: FaceMonitorAnalyzer,
    deepfake_detector: DeepfakeDetector,
) -> IdentityProfile:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RegistrationError("Nao foi possivel abrir o arquivo de video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RegistrationError("O video nao possui duracao ou quadros validos.")
    duration = frame_count / fps
    try:
        _validate_submitted_video_duration(duration)
    except RegistrationError:
        capture.release()
        raise

    sample_step = max(int(fps * 0.20), 1)
    embeddings: list[np.ndarray] = []
    face_crops: list[np.ndarray] = []
    aligned_face_crops: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    gray_crops: list[np.ndarray] = []
    orientations: set[str] = set()
    blink = BlinkDetector()
    frames_read = 0
    rejected_multiple_faces = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = frames_read
            frames_read += 1
            if frame_index % sample_step:
                continue

            faces = analyzer.detect_faces(frame)
            if len(faces) != 1:
                if len(faces) > 1:
                    rejected_multiple_faces += 1
                continue
            face = faces[0]
            quality = analyzer.evaluate_quality(frame, face)
            if not quality.adequate:
                continue
            crop = analyzer.face_crop(frame, face, margin_ratio=0.30)
            aligned_crop = analyzer.aligned_face_crop(frame, face)
            gray_crop = normalized_face_crop(frame, face)
            if crop is None or aligned_crop is None or gray_crop is None:
                continue

            embeddings.append(analyzer.embedding(face))
            face_crops.append(crop)
            aligned_face_crops.append(aligned_crop)
            landmarks.append(normalized_landmarks(face))
            gray_crops.append(gray_crop)
            orientations.add(ActiveChallenge.orientation(face))
            blink.update(face)
    finally:
        capture.release()

    if rejected_multiple_faces:
        raise RegistrationError("O video deve conter somente o candidato.")
    if len(embeddings) < 8:
        raise RegistrationError(
            "Poucos quadros faciais validos. Solicite um novo video com rosto proximo, nitido e bem iluminado."
        )
    if "CENTRO" not in orientations or not ({"ESQUERDA", "DIREITA"} & orientations):
        raise RegistrationError(
            "O video deve mostrar o rosto de frente e virado para pelo menos um dos lados."
        )
    if not blink.completed:
        raise RegistrationError("Nao foi possivel confirmar uma piscada no video enviado.")

    liveness = analyze_liveness(
        landmarks,
        gray_crops,
        analyzer.config.minimum_temporal_variation,
        analyzer.config.minimum_landmark_movement,
        analyzer.config.maximum_landmark_movement,
    )
    if not liveness.valid:
        raise RegistrationError("O arquivo nao apresentou variacao facial natural suficiente.")

    embeddings = _filter_consistent_embeddings(embeddings)
    if len(embeddings) < 8:
        raise RegistrationError("As amostras faciais do video nao sao consistentes entre si.")

    # Limita a inferencia e distribui as amostras por toda a duracao do arquivo.
    selected_indices = np.linspace(0, len(face_crops) - 1, min(12, len(face_crops)), dtype=int)
    detector_result = deepfake_detector.predict(
        [face_crops[index] for index in selected_indices],
        [aligned_face_crops[index] for index in selected_indices],
    )
    probability = detector_result.component_probabilities.get("dfdc")
    if probability is not None and probability >= 0.80:
        raise RegistrationError(
            f"O video apresentou alto indicio de manipulacao facial ({probability * 100:.1f}%)."
        )

    selected_embeddings = [
        embeddings[index]
        for index in np.linspace(0, len(embeddings) - 1, min(16, len(embeddings)), dtype=int)
    ]
    return IdentityProfile.create(
        name,
        selected_embeddings,
        source_count=len(embeddings),
        registration_method="submitted_video",
        source_deepfake_probability=probability,
        source_effort_probability=detector_result.effort_probability,
    )


def choose_reference_images() -> list[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilenames(
        title="Selecionar imagens de referencia",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos os arquivos", "*.*")],
    )
    root.destroy()
    return [Path(path) for path in selected]


def choose_reference_video() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Selecionar video enviado pelo candidato",
        filetypes=[
            ("Videos", "*.mp4 *.mov *.avi *.mkv *.webm"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None
