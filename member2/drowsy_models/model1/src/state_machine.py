from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .fusion_engine import FusionMetrics
from .types import FrameFeatures


class AlertState(Enum):
    UNKNOWN = "unknown"
    CALIBRATING = "calibrating"
    AWAKE = "awake"
    SUSPECT = "suspect"
    DROWSY = "drowsy"
    CRITICAL = "critical"


@dataclass(slots=True)
class StateMachineThresholds:
    suspect_yolo_probability: float = 0.65
    drowsy_yolo_probability: float = 0.75
    critical_yolo_probability: float = 0.90
    suspect_perclos: float = 0.15
    drowsy_perclos: float = 0.25
    long_eye_closure_seconds: float = 0.80
    critical_eye_closure_seconds: float = 1.50
    yawn_min_seconds: float = 1.20
    yawn_event_gap_seconds: float = 2.00
    head_down_delta_degrees: float = 15.0
    head_down_min_seconds: float = 1.50
    recovery_yolo_probability: float = 0.45
    recovery_seconds: float = 2.00
    face_missing_unknown_seconds: float = 0.70
    recent_yawn_seconds: float = 15.0

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "StateMachineThresholds":
        if not mapping:
            return cls()
        data = {field: mapping.get(field, getattr(cls(), field)) for field in cls.__dataclass_fields__}
        return cls(**data)


class StateMachine:
    def __init__(self, thresholds: StateMachineThresholds | dict | None = None) -> None:
        if isinstance(thresholds, dict):
            self.thresholds = StateMachineThresholds.from_mapping(thresholds)
        else:
            self.thresholds = thresholds or StateMachineThresholds()
        self.state = AlertState.UNKNOWN
        self._face_missing_since: float | None = None
        self._high_65_since: float | None = None
        self._high_75_since: float | None = None
        self._high_90_since: float | None = None
        self._eye_open_since: float | None = None
        self._low_prob_since: float | None = None
        self._yawn_active_since: float | None = None
        self._last_yawn_event_ts: float | None = None
        self._head_down_since: float | None = None
        self._state_since: float | None = None

    def update(
        self,
        features: FrameFeatures,
        metrics: FusionMetrics,
        timestamp: float | None = None,
        calibrating: bool = False,
    ) -> AlertState:
        now = features.timestamp if timestamp is None else timestamp
        if calibrating:
            self.state = AlertState.CALIBRATING
            self._state_since = now
            return self.state

        face_ok = features.face_valid and features.face_quality >= 0.25
        if face_ok:
            self._face_missing_since = None
        else:
            if self._face_missing_since is None:
                self._face_missing_since = now

        if face_ok and not features.eye_closed:
            if self._eye_open_since is None:
                self._eye_open_since = now
        else:
            self._eye_open_since = None

        if features.p_drowsy_ema is None:
            self._high_65_since = None
            self._high_75_since = None
            self._high_90_since = None
            self._low_prob_since = None
        else:
            if features.p_drowsy_ema < self.thresholds.suspect_yolo_probability:
                self._high_65_since = None
            elif self._high_65_since is None:
                self._high_65_since = now

            if features.p_drowsy_ema < self.thresholds.drowsy_yolo_probability:
                self._high_75_since = None
            elif self._high_75_since is None:
                self._high_75_since = now

            if features.p_drowsy_ema < self.thresholds.critical_yolo_probability:
                self._high_90_since = None
            elif self._high_90_since is None:
                self._high_90_since = now

            if features.p_drowsy_ema >= self.thresholds.recovery_yolo_probability:
                self._low_prob_since = None
            elif self._low_prob_since is None:
                self._low_prob_since = now

        if features.yawn_active:
            if self._yawn_active_since is None:
                self._yawn_active_since = now
        else:
            if self._yawn_active_since is not None:
                if now - self._yawn_active_since >= self.thresholds.yawn_min_seconds:
                    if self._last_yawn_event_ts is None or now - self._last_yawn_event_ts >= self.thresholds.yawn_event_gap_seconds:
                        self._last_yawn_event_ts = now
                self._yawn_active_since = None

        if features.head_down:
            if self._head_down_since is None:
                self._head_down_since = now
        else:
            self._head_down_since = None

        no_face_duration = 0.0 if self._face_missing_since is None else now - self._face_missing_since
        eye_open_duration = 0.0 if self._eye_open_since is None else now - self._eye_open_since
        yolo_65_duration = 0.0 if self._high_65_since is None else now - self._high_65_since
        yolo_75_duration = 0.0 if self._high_75_since is None else now - self._high_75_since
        yolo_90_duration = 0.0 if self._high_90_since is None else now - self._high_90_since
        low_prob_duration = 0.0 if self._low_prob_since is None else now - self._low_prob_since
        head_down_duration = 0.0 if self._head_down_since is None else now - self._head_down_since
        recent_yawn = self._last_yawn_event_ts is not None and now - self._last_yawn_event_ts <= self.thresholds.recent_yawn_seconds

        unknown = no_face_duration >= self.thresholds.face_missing_unknown_seconds
        if unknown:
            self.state = AlertState.UNKNOWN
            self._state_since = now
            return self.state

        critical = (
            features.current_eye_closure_seconds >= self.thresholds.critical_eye_closure_seconds
            or (
                yolo_90_duration >= 3.0
                and features.p_drowsy_ema is not None
                and features.p_drowsy_ema >= self.thresholds.critical_yolo_probability
                and (features.eye_closed or features.head_down)
            )
        )
        if critical:
            self.state = AlertState.CRITICAL
            self._state_since = now
            return self.state

        drowsy = (
            (
                yolo_75_duration >= 2.0
                and features.p_drowsy_ema is not None
                and features.p_drowsy_ema >= self.thresholds.drowsy_yolo_probability
                and (
                    metrics.perclos_30s >= 0.18
                    or features.current_eye_closure_seconds >= self.thresholds.long_eye_closure_seconds
                    or recent_yawn
                    or head_down_duration >= self.thresholds.head_down_min_seconds
                )
            )
            or (
                metrics.perclos_30s >= self.thresholds.drowsy_perclos
                and metrics.perclos_coverage_seconds >= 20.0
            )
            or (
                metrics.p_drowsy_high_ratio_5s >= 0.70
                and self.state != AlertState.UNKNOWN
            )
        )
        if drowsy:
            self.state = AlertState.DROWSY
            self._state_since = now
            return self.state

        suspect = (
            (yolo_65_duration >= 1.0 and features.p_drowsy_ema is not None and features.p_drowsy_ema >= self.thresholds.suspect_yolo_probability)
            or metrics.perclos_30s >= self.thresholds.suspect_perclos
            or recent_yawn
            or head_down_duration >= self.thresholds.head_down_min_seconds
        )

        recover = (
            eye_open_duration >= self.thresholds.recovery_seconds
            and low_prob_duration >= self.thresholds.recovery_seconds
            and not features.head_down
        )

        if self.state in {AlertState.DROWSY, AlertState.CRITICAL}:
            if recover:
                self.state = AlertState.AWAKE
                self._state_since = now
            elif suspect:
                self.state = AlertState.SUSPECT
                self._state_since = now
            return self.state

        if suspect:
            self.state = AlertState.SUSPECT
            self._state_since = now
        else:
            self.state = AlertState.AWAKE
            self._state_since = now
        return self.state
