from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.deepfake_detection.base import DeepfakeResult


class OnnxDeepfakeDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: tuple[int, int],
        model_name: str,
        fake_class_index: int = 1,
        normalization: str = "imagenet",
        resize_mode: str = "stretch",
        output_type: str = "auto",
    ):
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.model_name = model_name
        self.fake_class_index = fake_class_index
        self.normalization = normalization
        self.resize_mode = resize_mode
        self.output_type = output_type

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        if self.resize_mode == "letterbox":
            target_width, target_height = self.input_size
            height, width = crop.shape[:2]
            scale = min(target_width / width, target_height / height)
            resized = cv2.resize(crop, (max(1, int(width * scale)), max(1, int(height * scale))))
            image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y = (target_height - resized.shape[0]) // 2
            x = (target_width - resized.shape[1]) // 2
            image[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        else:
            image = cv2.resize(crop, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
        return np.transpose(image, (2, 0, 1))[None, ...]

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def predict(
        self,
        face_crops: list[np.ndarray],
        aligned_face_crops: list[np.ndarray] | None = None,
    ) -> DeepfakeResult:
        if not face_crops:
            return DeepfakeResult(None, True, self.model_name, "Sem recortes validos.")
        batch = np.concatenate([self._preprocess(crop) for crop in face_crops], axis=0)
        output = np.asarray(self.session.run(None, {self.input_name: batch})[0])
        if self.output_type == "logit":
            logits = np.clip(output.reshape(-1), -60.0, 60.0)
            probabilities = (1.0 / (1.0 + np.exp(-logits))).tolist()
        else:
            rows = output.reshape(len(face_crops), -1)
            probabilities = [
                float(self._softmax(row)[self.fake_class_index]) for row in rows
            ]
        probability = float(np.median(probabilities))
        return DeepfakeResult(
            probability,
            True,
            self.model_name,
            component_probabilities={"dfdc": probability},
        )
