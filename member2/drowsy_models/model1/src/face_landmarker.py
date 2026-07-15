from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .face_cropper import compute_face_bbox
from .types import FaceObservation


class FaceLandmarkerError(RuntimeError):
    """Raised when MediaPipe face landmarker setup fails."""


class MediaPipeFaceLandmarker:
    def __init__(
        self,
        model_path: str,
        mediapipe_config: dict | None = None,
        face_crop_config: dict | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.mediapipe_config = mediapipe_config or {}
        self.face_crop_config = face_crop_config or {}
        self._mp = None
        self._vision = None
        self._landmarker = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        if not self.model_path.exists():
            raise FaceLandmarkerError(f"MediaPipe model not found: {self.model_path}")
        with self.model_path.open("rb") as handle:
            prefix = handle.read(32)
        if prefix.startswith(b"PLACEHOLDER"):
            raise FaceLandmarkerError(
                "models/face_landmarker.task is still a placeholder. Replace it with the official MediaPipe task model."
            )

        try:  # pragma: no cover - depends on optional runtime dependency
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise FaceLandmarkerError("mediapipe is required for face landmark detection") from exc

        self._mp = mp
        self._vision = vision

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=int(self.mediapipe_config.get("num_faces", 1)),
            min_face_detection_confidence=float(
                self.mediapipe_config.get("min_face_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                self.mediapipe_config.get("min_face_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(self.mediapipe_config.get("min_tracking_confidence", 0.5)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> FaceObservation:
        if self._landmarker is None or self._mp is None:
            return FaceObservation(
                timestamp=timestamp,
                landmarks=None,
                blendshapes={},
                transform_matrix=None,
                bbox=None,
                face_valid=False,
            )

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(0, int(timestamp * 1000.0))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not getattr(result, "face_landmarks", None):
            return FaceObservation(
                timestamp=timestamp,
                landmarks=None,
                blendshapes={},
                transform_matrix=None,
                bbox=None,
                face_valid=False,
            )

        if len(result.face_landmarks) != 1:
            return FaceObservation(
                timestamp=timestamp,
                landmarks=None,
                blendshapes={},
                transform_matrix=None,
                bbox=None,
                face_valid=False,
            )

        landmarks = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        blendshapes: dict[str, float] = {}
        if getattr(result, "face_blendshapes", None):
            for category in result.face_blendshapes[0]:
                blendshapes[str(category.category_name)] = float(category.score)

        transform_matrix = None
        if getattr(result, "facial_transformation_matrixes", None):
            transform_matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)

        bbox = compute_face_bbox(
            landmarks=landmarks,
            image_width=frame_bgr.shape[1],
            image_height=frame_bgr.shape[0],
            horizontal_margin=float(self.face_crop_config.get("horizontal_margin", 0.20)),
            top_margin=float(self.face_crop_config.get("top_margin", 0.20)),
            bottom_margin=float(self.face_crop_config.get("bottom_margin", 0.25)),
            min_face_width_pixels=1,
            min_face_height_pixels=1,
        )
        return FaceObservation(
            timestamp=timestamp,
            landmarks=landmarks,
            blendshapes=blendshapes,
            transform_matrix=transform_matrix,
            bbox=bbox,
            face_valid=bbox is not None,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

