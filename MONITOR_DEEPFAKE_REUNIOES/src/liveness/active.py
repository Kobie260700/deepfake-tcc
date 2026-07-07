import random
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChallengeStatus:
    active: bool
    current_action: str | None
    progress: str
    remaining_seconds: float
    result: bool | None


class ActiveChallenge:
    def __init__(self, seconds_per_action: float = 7.0, stable_frames: int = 3):
        self.seconds_per_action = seconds_per_action
        self.stable_frames = stable_frames
        self.sequence = []
        self.index = 0
        self.correct_frames = 0
        self.action_started = 0.0
        self.active = False
        self.result = None
        self.open_ear_samples = []
        self.reference_ear = None
        self.blink_state = "OPEN"

    def start(self) -> None:
        first = random.SystemRandom().choice(("ESQUERDA", "DIREITA"))
        last = "DIREITA" if first == "ESQUERDA" else "ESQUERDA"
        self.sequence = [first, "CENTRO", "PISQUE", last]
        self.index = 0
        self.correct_frames = 0
        self.action_started = time.monotonic()
        self.active = True
        self.result = None
        self._reset_blink()

    def cancel(self) -> None:
        self.active = False
        self.result = False

    def _reset_blink(self) -> None:
        self.open_ear_samples = []
        self.reference_ear = None
        self.blink_state = "OPEN"

    @staticmethod
    def orientation(face) -> str:
        points = np.asarray(face.kps, dtype=np.float32)
        left_eye, right_eye, nose = points[0], points[1], points[2]
        eyes_center = (left_eye[0] + right_eye[0]) / 2
        eyes_distance = max(abs(right_eye[0] - left_eye[0]), 1.0)
        displacement = float((nose[0] - eyes_center) / eyes_distance)
        if displacement <= -0.13:
            return "DIREITA"
        if displacement >= 0.13:
            return "ESQUERDA"
        if abs(displacement) <= 0.08:
            return "CENTRO"
        return "TRANSICAO"

    @staticmethod
    def _ear(eye: np.ndarray) -> float:
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = max(np.linalg.norm(eye[0] - eye[3]), 1e-6)
        return float((vertical_1 + vertical_2) / (2 * horizontal))

    def _blink_completed(self, face) -> bool:
        points = getattr(face, "landmark_3d_68", None)
        if points is None:
            return False
        points = np.asarray(points, dtype=np.float32)[:, :2]
        ear = (self._ear(points[36:42]) + self._ear(points[42:48])) / 2

        if self.reference_ear is None:
            self.open_ear_samples.append(ear)
            if len(self.open_ear_samples) >= 4:
                self.reference_ear = float(np.median(self.open_ear_samples))
            return False

        closed = ear < self.reference_ear * 0.72
        if self.blink_state == "OPEN" and closed:
            self.blink_state = "CLOSED"
        elif self.blink_state == "CLOSED" and not closed:
            self.blink_state = "COMPLETED"
            return True
        return self.blink_state == "COMPLETED"

    def update(self, face, now: float | None = None) -> ChallengeStatus:
        now = now or time.monotonic()
        if not self.active:
            return self.status(now)

        if now - self.action_started > self.seconds_per_action:
            self.active = False
            self.result = False
            return self.status(now)

        action = self.sequence[self.index]
        matched = self._blink_completed(face) if action == "PISQUE" else self.orientation(face) == action
        self.correct_frames = self.correct_frames + 1 if matched else 0

        if self.correct_frames >= self.stable_frames:
            self.index += 1
            self.correct_frames = 0
            if self.index >= len(self.sequence):
                self.active = False
                self.result = True
            else:
                self.action_started = now
                self._reset_blink()
        return self.status(now)

    def tick(self, now: float | None = None) -> ChallengeStatus:
        now = now or time.monotonic()
        if self.active and now - self.action_started > self.seconds_per_action:
            self.active = False
            self.result = False
        return self.status(now)

    def status(self, now: float | None = None) -> ChallengeStatus:
        now = now or time.monotonic()
        current = self.sequence[self.index] if self.active and self.index < len(self.sequence) else None
        remaining = max(0.0, self.seconds_per_action - (now - self.action_started)) if self.active else 0.0
        return ChallengeStatus(
            self.active,
            current,
            f"{min(self.index + 1, len(self.sequence))}/{len(self.sequence)}" if self.sequence else "0/0",
            remaining,
            self.result,
        )
