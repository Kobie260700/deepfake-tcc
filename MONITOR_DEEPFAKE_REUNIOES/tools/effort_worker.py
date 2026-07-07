"""Persistent inference worker for the official Effort implementation.

The worker intentionally lives outside the monitor environment because CLIP-L/14
and CUDA-enabled PyTorch have a much larger dependency footprint.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = args.repository.resolve() / "DeepfakeBench" / "training"
    sys.path.insert(0, str(training_dir))

    # The official implementation resolves CLIP through ../models--openai--...
    # relative to the current directory. Use a dedicated runtime directory so
    # the downloaded model can remain outside the cloned repository.
    runtime_dir = args.base_model.resolve().parent / "effort-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    expected_model = runtime_dir.parent / "models--openai--clip-vit-large-patch14"
    if expected_model.resolve() != args.base_model.resolve():
        raise RuntimeError(f"Diretorio CLIP inesperado: {args.base_model}")
    os.chdir(runtime_dir)

    import cv2
    import numpy as np
    import torch
    import yaml
    from detectors.effort_detector import EffortDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = training_dir / "config" / "detector" / "effort.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = EffortDetector(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    inference_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=inference_dtype)
    model.eval()

    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=device,
        dtype=inference_dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=device,
        dtype=inference_dtype,
    ).view(1, 3, 1, 1)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            probabilities = []
            for image_path in request.get("paths", []):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
                    device=device,
                    dtype=inference_dtype,
                )
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = (tensor - mean) / std
                with torch.inference_mode():
                    prediction = model({"image": tensor}, inference=True)
                probabilities.append(float(prediction["prob"].item()))
            if not probabilities:
                raise RuntimeError("Nenhum recorte facial valido recebido.")
            response = {"probabilities": probabilities, "device": str(device)}
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
