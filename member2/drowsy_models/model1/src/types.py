from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass(slots=True)
class FaceObservation:
    timestamp: float
    landmarks: np.ndarray | None
    blendshapes: dict[str, float]
    transform_matrix: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    face_valid: bool


@dataclass(slots=True)
class ClassificationObservation:
    timestamp: float
    p_drowsy: float
    p_awake: float
    latency_ms: float
    valid: bool


@dataclass(slots=True)
class FrameFeatures:
    timestamp: float

    face_valid: bool
    face_quality: float

    eye_closed_score: float | None
    eye_closed: bool
    current_eye_closure_seconds: float

    mouth_open_score: float | None
    yawn_active: bool

    yaw: float | None
    pitch: float | None
    roll: float | None
    head_down: bool

    p_drowsy_raw: float | None
    p_drowsy_ema: float | None
    yolo_result_stale: bool
    eye_blink_left: float | None = None
    eye_blink_right: float | None = None
    yolo_latency_ms: float | None = None
