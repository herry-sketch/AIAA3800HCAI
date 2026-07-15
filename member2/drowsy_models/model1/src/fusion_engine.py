from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .temporal_buffer import TemporalBuffer


@dataclass(slots=True)
class FusionMetrics:
    timestamp: float
    perclos_30s: float
    perclos_coverage_seconds: float
    p_drowsy_mean_5s: float | None
    p_drowsy_high_ratio_5s: float
    recent_yawn_event: bool
    current_eye_closure_seconds: float
    current_head_down_duration: float
    current_p_drowsy_ema: float | None
    current_p_drowsy_raw: float | None
    current_face_valid: bool
    current_face_quality: float
    current_yolo_stale: bool
    risk_score: float


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class FusionEngine:
    def __init__(
        self,
        short_window_seconds: float = 5.0,
        long_window_seconds: float = 30.0,
        yawn_min_seconds: float = 1.2,
    ) -> None:
        self.short_window_seconds = short_window_seconds
        self.long_window_seconds = long_window_seconds
        self.yawn_min_seconds = yawn_min_seconds

    def compute_metrics(
        self,
        buffer: TemporalBuffer,
        timestamp: float | None = None,
    ) -> FusionMetrics:
        latest = buffer.latest()
        if latest is None:
            now = 0.0 if timestamp is None else timestamp
            return FusionMetrics(
                timestamp=now,
                perclos_30s=0.0,
                perclos_coverage_seconds=0.0,
                p_drowsy_mean_5s=None,
                p_drowsy_high_ratio_5s=0.0,
                recent_yawn_event=False,
                current_eye_closure_seconds=0.0,
                current_head_down_duration=0.0,
                current_p_drowsy_ema=None,
                current_p_drowsy_raw=None,
                current_face_valid=False,
                current_face_quality=0.0,
                current_yolo_stale=True,
                risk_score=0.0,
            )

        now = latest.timestamp if timestamp is None else timestamp
        long_window = buffer.get_window(self.long_window_seconds)
        short_window = buffer.get_window(self.short_window_seconds)

        perclos_coverage = 0.0
        perclos_closed = 0.0
        short_total = 0.0
        short_sum = 0.0
        short_high = 0.0

        def interval_end(index: int, samples: list) -> float:
            if index + 1 < len(samples):
                return min(samples[index + 1].timestamp, now)
            return now

        for idx, sample in enumerate(long_window):
            start = sample.timestamp
            end = interval_end(idx, long_window)
            dt = max(0.0, end - start)
            if dt <= 0:
                continue
            if sample.face_valid and sample.face_quality > 0 and sample.eye_closed_score is not None:
                perclos_coverage += dt
                if sample.eye_closed:
                    perclos_closed += dt

        for idx, sample in enumerate(short_window):
            start = sample.timestamp
            end = interval_end(idx, short_window)
            dt = max(0.0, end - start)
            if dt <= 0:
                continue
            if sample.p_drowsy_ema is not None and not sample.yolo_result_stale and sample.face_valid:
                short_total += dt
                short_sum += sample.p_drowsy_ema * dt
                if sample.p_drowsy_ema >= 0.70:
                    short_high += dt

        p_drowsy_mean_5s = short_sum / short_total if short_total > 0 else None
        p_drowsy_high_ratio_5s = short_high / short_total if short_total > 0 else 0.0
        perclos_30s = perclos_closed / perclos_coverage if perclos_coverage > 0 else 0.0

        recent_yawn_event = False
        run_start: float | None = None
        run_end: float | None = None
        for idx, sample in enumerate(long_window):
            end = interval_end(idx, long_window)
            if sample.face_valid and sample.yawn_active:
                if run_start is None:
                    run_start = sample.timestamp
                run_end = end
            else:
                if run_start is not None and run_end is not None:
                    if run_end - run_start >= self.yawn_min_seconds and now - run_end <= 15.0:
                        recent_yawn_event = True
                        break
                run_start = None
                run_end = None
        if not recent_yawn_event and run_start is not None and run_end is not None:
            if run_end - run_start >= self.yawn_min_seconds and now - run_end <= 15.0:
                recent_yawn_event = True

        current = latest
        current_head_down_duration = 0.0
        for sample in reversed(long_window):
            if sample.head_down and sample.face_valid:
                current_head_down_duration = now - sample.timestamp
            else:
                break

        risk_score = (
            0.45 * (current.p_drowsy_ema or 0.0)
            + 0.30 * _clip(perclos_30s / 0.30, 0.0, 1.0)
            + 0.15 * _clip(current.current_eye_closure_seconds / 1.5, 0.0, 1.0)
            + 0.05 * float(recent_yawn_event)
            + 0.05 * float(current.head_down)
        )

        return FusionMetrics(
            timestamp=now,
            perclos_30s=perclos_30s,
            perclos_coverage_seconds=perclos_coverage,
            p_drowsy_mean_5s=p_drowsy_mean_5s,
            p_drowsy_high_ratio_5s=p_drowsy_high_ratio_5s,
            recent_yawn_event=recent_yawn_event,
            current_eye_closure_seconds=current.current_eye_closure_seconds,
            current_head_down_duration=current_head_down_duration,
            current_p_drowsy_ema=current.p_drowsy_ema,
            current_p_drowsy_raw=current.p_drowsy_raw,
            current_face_valid=current.face_valid,
            current_face_quality=current.face_quality,
            current_yolo_stale=current.yolo_result_stale,
            risk_score=risk_score,
        )
