from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""


def _get_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency at test time
        raise RuntimeError("opencv-python is required for camera access") from exc
    return cv2


@dataclass
class CameraStream:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    retry_count: int = 3
    retry_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        self._cv2 = _get_cv2()
        self._capture = None

    def open(self) -> None:
        self._capture = self._cv2.VideoCapture(self.index)
        if not self._capture.isOpened():
            raise CameraError(f"Failed to open camera index {self.index}")
        self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(self._cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._capture is None:
            raise CameraError("Camera has not been opened")
        for attempt in range(self.retry_count):
            ok, frame = self._capture.read()
            timestamp = time.monotonic()
            if ok and frame is not None:
                return True, frame, timestamp
            time.sleep(self.retry_delay_seconds)
        return False, None, time.monotonic()

    def make_display_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.mirror:
            return frame.copy()
        return np.ascontiguousarray(frame[:, ::-1])

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def close_windows(self) -> None:
        self._cv2.destroyAllWindows()

