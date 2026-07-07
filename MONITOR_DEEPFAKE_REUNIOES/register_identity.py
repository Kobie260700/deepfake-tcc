import argparse
from pathlib import Path

from src.config import MonitorConfig
from src.face_tracking.analyzer import FaceMonitorAnalyzer
from src.deepfake_detection.factory import load_detector
from src.identity.registration import (
    choose_reference_video,
    create_profile_from_images,
    create_profile_from_video_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadastro de referencia facial do candidato.")
    parser.add_argument("--name", required=True, help="Nome da identidade cadastrada.")
    parser.add_argument("--images", nargs="*", type=Path, help="Imagens de referencia.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Video de referencia enviado pelo candidato.",
    )
    args = parser.parse_args()

    config = MonitorConfig.load(PROJECT_ROOT / "config" / "monitor.json")
    analyzer = FaceMonitorAnalyzer(config)
    if args.video:
        profile = create_profile_from_video_file(
            args.name,
            args.video,
            analyzer,
            load_detector(PROJECT_ROOT / "models"),
        )
    elif args.images:
        profile = create_profile_from_images(args.name, args.images, analyzer)
    else:
        video_path = choose_reference_video()
        if video_path is not None:
            profile = create_profile_from_video_file(
                args.name,
                video_path,
                analyzer,
                load_detector(PROJECT_ROOT / "models"),
            )
        else:
            raise SystemExit("Nenhum video foi selecionado.")
    profile_dir = profile.save(PROJECT_ROOT / "profiles")
    print(f"[SUCESSO] Perfil criado: {profile_dir}")
    print(f"[INFO] Embeddings validos: {len(profile.embeddings)}")
    if profile.source_deepfake_probability is not None:
        print(
            "[INFO] Indicio de manipulacao no video de referencia: "
            f"{profile.source_deepfake_probability * 100:.1f}%"
        )
    if profile.source_effort_probability is not None:
        print(
            "[INFO] Baseline Effort da referencia: "
            f"{profile.source_effort_probability * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
