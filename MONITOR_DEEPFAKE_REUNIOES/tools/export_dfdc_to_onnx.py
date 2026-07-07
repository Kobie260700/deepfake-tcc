import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn
import timm


class DeepFakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("tf_efficientnet_b7_ns", pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(2560, 1)

    def forward(self, inputs):
        features = self.encoder.forward_features(inputs)
        features = self.avg_pool(features).flatten(1)
        return self.fc(self.dropout(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta o detector DFDC EfficientNet-B7 para ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = DeepFakeClassifier()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {re.sub(r"^module\.", "", key): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [key for key in missing if not key.startswith("encoder.classifier")]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompativel. Ausentes={meaningful_missing}; inesperadas={unexpected}"
        )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 380, 380, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logit"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        dynamo=False,
    )

    config = {
        "name": "DFDC EfficientNet-B7 (selimsef)",
        "source": "https://github.com/selimsef/dfdc_deepfake_challenge",
        "license": "MIT",
        "file": args.output.name,
        "input_size": [380, 380],
        "fake_class_index": 0,
        "normalization": "imagenet",
        "resize_mode": "letterbox",
        "output_type": "logit",
    }
    (args.output.parent / "detector.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Modelo exportado: {args.output}")


if __name__ == "__main__":
    main()
