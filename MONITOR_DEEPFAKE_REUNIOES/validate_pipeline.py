import argparse
from pathlib import Path

import cv2

from src.config import MonitorConfig
from src.deepfake_detection.factory import load_detector
from src.face_tracking.analyzer import FaceMonitorAnalyzer
from src.identity.profile import IdentityProfile, IdentityVerifier


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o pipeline em imagens estaticas.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    config = MonitorConfig.load(PROJECT_ROOT / "config" / "monitor.json")
    analyzer = FaceMonitorAnalyzer(config)
    verifier = IdentityVerifier(IdentityProfile.load(args.profile), config.identity_distance_threshold)
    detector = load_detector(PROJECT_ROOT / "models")

    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path}: arquivo invalido")
            continue
        face = analyzer.detect_primary_face(frame)
        if face is None:
            print(f"{path}: rosto nao detectado")
            continue
        identity = verifier.verify([analyzer.embedding(face)])
        crop = analyzer.face_crop(frame, face, margin_ratio=0.30)
        deepfake = detector.predict([crop] if crop is not None else [])
        probability = "indisponivel" if deepfake.probability is None else f"{deepfake.probability:.4f}"
        print(
            f"{path.name}: distancia={identity.distance:.4f}; "
            f"compativel={identity.compatible}; deepfake={probability}"
        )


if __name__ == "__main__":
    main()
