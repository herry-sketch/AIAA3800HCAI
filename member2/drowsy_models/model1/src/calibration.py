from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

import numpy as np

from .types import FrameFeatures


@dataclass(slots=True)
class CalibrationResult:
    enabled: bool
    complete: bool
    quality_ok: bool
    quality_message: str
    eye_closed_threshold: float
    yawn_threshold: float
    baseline_pitch: float
    sample_count: int
    valid_seconds: float
    p_drowsy_samples: list[float] = field(default_factory=list)


@dataclass
class CalibrationCollector:
    duration_seconds: float
    default_eye_closed_threshold: float = 0.65
    default_yawn_threshold: float = 0.70
    min_valid_seconds: float = 5.0
    max_abs_yaw_for_sample: float = 35.0
    max_abs_pitch_for_sample: float = 40.0

    _eye_scores: list[float] = field(default_factory=list, init=False)
    _mouth_scores: list[float] = field(default_factory=list, init=False)
    _pitch_scores: list[float] = field(default_factory=list, init=False)
    _p_drowsy_scores: list[float] = field(default_factory=list, init=False)
    _valid_seconds: float = field(default=0.0, init=False)
    _last_timestamp: float | None = field(default=None, init=False)
    _last_was_valid: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._eye_scores.clear()
        self._mouth_scores.clear()
        self._pitch_scores.clear()
        self._p_drowsy_scores.clear()
        self._valid_seconds = 0.0
        self._last_timestamp = None
        self._last_was_valid = False

    def update(self, features: FrameFeatures) -> None:
        timestamp = features.timestamp
        if self._last_timestamp is not None and self._last_was_valid:
            self._valid_seconds += max(0.0, timestamp - self._last_timestamp)

        is_valid = (
            features.face_valid
            and features.face_quality >= 0.3
            and (features.yaw is None or abs(features.yaw) <= self.max_abs_yaw_for_sample)
            and (features.pitch is None or abs(features.pitch) <= self.max_abs_pitch_for_sample)
        )

        if is_valid:
            if features.eye_closed_score is not None:
                self._eye_scores.append(features.eye_closed_score)
            if features.mouth_open_score is not None:
                self._mouth_scores.append(features.mouth_open_score)
            if features.pitch is not None:
                self._pitch_scores.append(features.pitch)
            if features.p_drowsy_raw is not None:
                self._p_drowsy_scores.append(features.p_drowsy_raw)

        self._last_timestamp = timestamp
        self._last_was_valid = is_valid

    def ready(self) -> bool:
        return self._valid_seconds >= self.duration_seconds

    def finalize(self) -> CalibrationResult:
        if self._valid_seconds < self.min_valid_seconds or len(self._eye_scores) < 5:
            return CalibrationResult(
                enabled=True,
                complete=False,
                quality_ok=False,
                quality_message="校准质量不足",
                eye_closed_threshold=self.default_eye_closed_threshold,
                yawn_threshold=self.default_yawn_threshold,
                baseline_pitch=0.0,
                sample_count=len(self._eye_scores),
                valid_seconds=self._valid_seconds,
                p_drowsy_samples=list(self._p_drowsy_scores),
            )

        eye_open_p95 = float(np.percentile(np.asarray(self._eye_scores, dtype=float), 95))
        mouth_neutral_p95 = float(np.percentile(np.asarray(self._mouth_scores or [0.0], dtype=float), 95))
        baseline_pitch = float(median(self._pitch_scores or [0.0]))

        eye_closed_threshold = float(np.clip(eye_open_p95 + 0.25, 0.55, 0.85))
        yawn_threshold = float(np.clip(mouth_neutral_p95 + 0.30, 0.60, 0.90))

        return CalibrationResult(
            enabled=True,
            complete=True,
            quality_ok=True,
            quality_message="校准完成",
            eye_closed_threshold=eye_closed_threshold,
            yawn_threshold=yawn_threshold,
            baseline_pitch=baseline_pitch,
            sample_count=len(self._eye_scores),
            valid_seconds=self._valid_seconds,
            p_drowsy_samples=list(self._p_drowsy_scores),
        )

