from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt

import numpy as np

from .calibration import CalibrationResult
from .types import ClassificationObservation, FaceObservation, FrameFeatures, HeadPose


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def _blendshape_lookup(blendshapes: dict[str, float], key: str, default: float = 0.0) -> float:
    target = _normalize_name(key)
    for candidate, value in blendshapes.items():
        if _normalize_name(candidate) == target:
            return float(value)
    return float(default)


def rotation_matrix_to_euler(matrix: np.ndarray) -> HeadPose:
    mat = np.asarray(matrix, dtype=float)
    if mat.shape == (4, 4):
        mat = mat[:3, :3]
    if mat.shape != (3, 3):
        raise ValueError("Expected 3x3 or 4x4 rotation matrix")

    sy = sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = atan2(mat[2, 1], mat[2, 2])
        pitch = atan2(-mat[2, 0], sy)
        yaw = atan2(mat[1, 0], mat[0, 0])
    else:
        roll = atan2(-mat[1, 2], mat[1, 1])
        pitch = atan2(-mat[2, 0], sy)
        yaw = 0.0
    return HeadPose(yaw=degrees(yaw), pitch=degrees(pitch), roll=degrees(roll))


@dataclass
class FeatureExtractorConfig:
    yolo_ema_alpha: float = 0.20
    yolo_stale_seconds: float = 0.50
    yolo_max_age_seconds: float = 2.0
    head_down_delta_degrees: float = 15.0
    head_down_min_seconds: float = 1.50
    max_abs_yaw_for_valid_classification: float = 35.0
    max_abs_pitch_for_valid_classification: float = 40.0
    min_face_width_pixels: int = 120
    min_face_height_pixels: int = 120


class FeatureExtractor:
    def __init__(self, config: FeatureExtractorConfig | dict | None = None) -> None:
        if isinstance(config, dict):
            self.config = FeatureExtractorConfig(**config)
        else:
            self.config = config or FeatureExtractorConfig()
        self._previous_features: FrameFeatures | None = None
        self._last_yolo_timestamp: float | None = None
        self._p_drowsy_ema: float | None = None

    def update(
        self,
        face_observation: FaceObservation,
        yolo_observation: ClassificationObservation | None,
        calibration: CalibrationResult,
        timestamp: float | None = None,
    ) -> FrameFeatures:
        timestamp = face_observation.timestamp if timestamp is None else timestamp
        blendshapes = face_observation.blendshapes or {}

        eye_left = _blendshape_lookup(blendshapes, "eyeBlinkLeft")
        eye_right = _blendshape_lookup(blendshapes, "eyeBlinkRight")
        eye_closed_score = min(eye_left, eye_right) if blendshapes else None
        mouth_open_score = _blendshape_lookup(blendshapes, "jawOpen") if blendshapes else None

        yaw = pitch = roll = None
        if face_observation.transform_matrix is not None:
            pose = rotation_matrix_to_euler(face_observation.transform_matrix)
            yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

        face_valid = bool(face_observation.face_valid and face_observation.landmarks is not None and face_observation.bbox is not None)
        face_quality = 0.0
        if face_valid and face_observation.bbox is not None:
            x1, y1, x2, y2 = face_observation.bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            size_factor = min(width / self.config.min_face_width_pixels, height / self.config.min_face_height_pixels)
            face_quality = float(max(0.0, min(1.0, size_factor)))
            if yaw is not None and abs(yaw) > self.config.max_abs_yaw_for_valid_classification:
                face_quality *= 0.5
            if pitch is not None and abs(pitch) > self.config.max_abs_pitch_for_valid_classification:
                face_quality *= 0.5

        if not face_valid:
            features = FrameFeatures(
                timestamp=timestamp,
                face_valid=False,
                face_quality=0.0,
                eye_closed_score=eye_closed_score,
                eye_closed=False,
                current_eye_closure_seconds=0.0,
                mouth_open_score=mouth_open_score,
                yawn_active=False,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                head_down=False,
                p_drowsy_raw=None,
                p_drowsy_ema=None,
                yolo_result_stale=True,
                eye_blink_left=eye_left if blendshapes else None,
                eye_blink_right=eye_right if blendshapes else None,
                yolo_latency_ms=None,
            )
            self._previous_features = features
            return features

        eye_closed = bool(eye_closed_score is not None and eye_closed_score >= calibration.eye_closed_threshold)
        mouth_valid = mouth_open_score is not None
        yawn_active = bool(mouth_valid and mouth_open_score >= calibration.yawn_threshold)

        if self._previous_features is not None and self._previous_features.eye_closed and eye_closed:
            current_eye_closure_seconds = max(0.0, timestamp - self._previous_features.timestamp) + self._previous_features.current_eye_closure_seconds
        elif eye_closed:
            current_eye_closure_seconds = 0.0
        else:
            current_eye_closure_seconds = 0.0

        head_down = False
        if pitch is not None:
            relative_pitch = pitch - calibration.baseline_pitch
            if (
                abs(yaw or 0.0) <= self.config.max_abs_yaw_for_valid_classification
                and abs(pitch) <= self.config.max_abs_pitch_for_valid_classification
                and relative_pitch > self.config.head_down_delta_degrees
            ):
                head_down = True

        p_drowsy_raw = None
        yolo_result_stale = True
        p_drowsy_ema = self._p_drowsy_ema
        yolo_latency_ms = None
        if yolo_observation is not None and yolo_observation.valid:
            age = max(0.0, timestamp - yolo_observation.timestamp)
            if age <= self.config.yolo_max_age_seconds:
                p_drowsy_raw = yolo_observation.p_drowsy
                yolo_result_stale = age > self.config.yolo_stale_seconds
                yolo_latency_ms = yolo_observation.latency_ms
                if self._last_yolo_timestamp is None or yolo_observation.timestamp > self._last_yolo_timestamp:
                    if self._p_drowsy_ema is None:
                        self._p_drowsy_ema = yolo_observation.p_drowsy
                    else:
                        alpha = self.config.yolo_ema_alpha
                        self._p_drowsy_ema = alpha * yolo_observation.p_drowsy + (1.0 - alpha) * self._p_drowsy_ema
                    self._last_yolo_timestamp = yolo_observation.timestamp
                p_drowsy_ema = self._p_drowsy_ema
            else:
                self._p_drowsy_ema = None
                self._last_yolo_timestamp = None
                p_drowsy_ema = None
        elif self._last_yolo_timestamp is not None:
            age = max(0.0, timestamp - self._last_yolo_timestamp)
            if age <= self.config.yolo_max_age_seconds:
                p_drowsy_ema = self._p_drowsy_ema
                yolo_result_stale = age > self.config.yolo_stale_seconds
            else:
                self._p_drowsy_ema = None
                self._last_yolo_timestamp = None
                p_drowsy_ema = None
                p_drowsy_raw = None
                yolo_result_stale = True

        features = FrameFeatures(
            timestamp=timestamp,
            face_valid=face_valid,
            face_quality=face_quality,
            eye_closed_score=eye_closed_score,
            eye_closed=eye_closed,
            current_eye_closure_seconds=current_eye_closure_seconds,
            mouth_open_score=mouth_open_score,
            yawn_active=yawn_active,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            head_down=head_down,
            p_drowsy_raw=p_drowsy_raw,
            p_drowsy_ema=p_drowsy_ema,
            yolo_result_stale=yolo_result_stale,
            eye_blink_left=eye_left if blendshapes else None,
            eye_blink_right=eye_right if blendshapes else None,
            yolo_latency_ms=yolo_latency_ms,
        )
        self._previous_features = features
        return features
