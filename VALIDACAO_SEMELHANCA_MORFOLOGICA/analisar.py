from __future__ import annotations

import argparse
from pathlib import Path

from morfologia import MorphologyAnalyzer, analyze_set, save_profile_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai uma ficha morfologica de imagens e videos de um rosto."
    )
    parser.add_argument("--entrada", nargs="+", required=True, type=Path)
    parser.add_argument("--saida", type=Path, default=Path("resultados"))
    parser.add_argument("--max-amostras", type=int, default=40)
    parser.add_argument("--intervalo-video", type=float, default=0.5)
    return parser.parse_args()


def run(
    files: list[Path],
    output: Path,
    max_samples: int = 40,
    interval: float = 0.5,
) -> dict:
    analyzer = MorphologyAnalyzer()
    samples, rejected = analyzer.collect(files, max_samples, interval)
    profile = analyze_set("Rosto de referencia", files, samples, rejected)
    save_profile_reports(output, profile, samples)
    return {
        "samples": profile.samples,
        "rejected_frames": profile.rejected_frames,
        "ratios": profile.ratios,
        "internal_shape_rmse": profile.internal_shape_rmse,
    }


def main() -> None:
    args = parse_args()
    result = run(args.entrada, args.saida, args.max_amostras, args.intervalo_video)
    print(f"Amostras validas: {result['samples']}")
    print(f"Relatorios: {args.saida.resolve()}")


if __name__ == "__main__":
    main()

