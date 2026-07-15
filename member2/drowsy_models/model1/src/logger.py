from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .fusion_engine import FusionMetrics
from .state_machine import AlertState
from .types import FrameFeatures


class SessionLogger:
    def __init__(self, enabled: bool, output_directory: str) -> None:
        self.enabled = enabled
        self.output_directory = Path(output_directory)
        self.path: Path | None = None
        self._handle = None
        if self.enabled:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.jsonl")
            self.path = self.output_directory / filename
            self._handle = self.path.open("a", encoding="utf-8")

    def toggle(self) -> None:
        self.enabled = not self.enabled
        if not self.enabled and self._handle is not None:
            self._handle.flush()

    def log(
        self,
        state: AlertState,
        features: FrameFeatures,
        metrics: FusionMetrics,
        camera_fps: float,
        yolo_latency_ms: float | None,
    ) -> None:
        if not self.enabled or self._handle is None:
            return
        record = {
            "timestamp": features.timestamp,
            "state": state.value,
            "face_valid": features.face_valid,
            "face_quality": features.face_quality,
            "eye_blink_left": features.eye_blink_left,
            "eye_blink_right": features.eye_blink_right,
            "eye_closed_score": features.eye_closed_score,
            "eye_closed": features.eye_closed,
            "eye_closure_seconds": features.current_eye_closure_seconds,
            "mouth_open_score": features.mouth_open_score,
            "yawn_active": features.yawn_active,
            "yaw": features.yaw,
            "pitch": features.pitch,
            "roll": features.roll,
            "head_down": features.head_down,
            "p_drowsy_raw": features.p_drowsy_raw,
            "p_drowsy_ema": features.p_drowsy_ema,
            "p_drowsy_mean_5s": metrics.p_drowsy_mean_5s,
            "p_drowsy_high_ratio_5s": metrics.p_drowsy_high_ratio_5s,
            "perclos_30s": metrics.perclos_30s,
            "risk_score": metrics.risk_score,
            "yolo_result_stale": features.yolo_result_stale,
            "camera_fps": camera_fps,
            "yolo_latency_ms": yolo_latency_ms,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
