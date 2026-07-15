from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .fusion_engine import FusionMetrics
from .state_machine import AlertState
from .types import FaceObservation, FrameFeatures


def _get_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency at test time
        raise RuntimeError("opencv-python is required for rendering the UI") from exc
    return cv2


def _state_color(state: AlertState) -> tuple[int, int, int]:
    if state == AlertState.AWAKE:
        return (0, 180, 0)
    if state == AlertState.SUSPECT:
        return (0, 220, 220)
    if state == AlertState.DROWSY:
        return (0, 0, 255)
    if state == AlertState.CRITICAL:
        blink = int(time.monotonic() * 4.0) % 2
        return (0, 0, 255) if blink else (0, 0, 120)
    if state == AlertState.CALIBRATING:
        return (255, 120, 0)
    return (150, 150, 150)


def mirror_bbox(bbox: tuple[int, int, int, int], image_width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return image_width - x2, y1, image_width - x1, y2


@dataclass(slots=True)
class YoloBadge:
    label: str
    subtitle: str
    color: tuple[int, int, int]


def build_yolo_badge(features: FrameFeatures) -> YoloBadge:
    if not features.face_valid:
        return YoloBadge("N/A", "no face", (150, 150, 150))
    if features.yolo_result_stale:
        return YoloBadge("STALE", "waiting for fresh result", (160, 160, 160))
    if features.p_drowsy_raw is None:
        return YoloBadge("N/A", "no result", (150, 150, 150))

    p_drowsy = float(features.p_drowsy_raw)
    label = "DROWSY" if p_drowsy >= 0.5 else "AWAKE"
    confidence = max(p_drowsy, 1.0 - p_drowsy)
    color = (0, 0, 255) if label == "DROWSY" else (0, 180, 0)
    return YoloBadge(label, f"confidence {confidence * 100:.0f}%", color)


def _format_yolo_value(value: float | None, features: FrameFeatures, digits: int = 2) -> str:
    if not features.face_valid:
        return "N/A"
    if features.yolo_result_stale:
        return "stale"
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _draw_yolo_badge(frame: np.ndarray, badge: YoloBadge) -> None:
    cv2 = _get_cv2()
    height, width = frame.shape[:2]
    panel_w = min(360, max(260, width // 4))
    panel_h = 160
    x2 = width - 20
    y1 = 20
    x1 = max(20, x2 - panel_w)
    y2 = min(height - 20, y1 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), thickness=-1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), badge.color, thickness=3)

    cv2.putText(
        frame,
        "YOLO",
        (x1 + 18, y1 + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 1.55 if badge.label not in {"N/A", "STALE"} else 1.35
    label_thickness = 4
    label_size, _ = cv2.getTextSize(badge.label, label_font, label_scale, label_thickness)
    label_x = x1 + max(18, (panel_w - label_size[0]) // 2)
    label_y = y1 + 112
    cv2.putText(
        frame,
        badge.label,
        (label_x, label_y),
        label_font,
        label_scale,
        badge.color,
        label_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        badge.subtitle,
        (x1 + 18, y2 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    state: AlertState,
    face_observation: FaceObservation,
    features: FrameFeatures,
    metrics: FusionMetrics,
    camera_fps: float,
    yolo_fps: float,
    calibration_text: str = "",
    debug: bool = True,
    mirror: bool = False,
) -> np.ndarray:
    cv2 = _get_cv2()
    output = frame.copy()
    color = _state_color(state)
    badge = build_yolo_badge(features)

    bbox = face_observation.bbox
    if bbox is not None:
        draw_bbox = mirror_bbox(bbox, output.shape[1]) if mirror else bbox
        cv2.rectangle(output, draw_bbox[:2], draw_bbox[2:], color, 2)

    def fmt(value: float | None, digits: int = 2) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"

    lines = [
        f"State: {state.value}",
        f"face_valid: {features.face_valid}",
        f"face_quality: {fmt(features.face_quality)}",
        f"eyeBlinkLeft: {fmt(features.eye_blink_left)}",
        f"eyeBlinkRight: {fmt(features.eye_blink_right)}",
        f"eye_closed_score: {fmt(features.eye_closed_score)}",
        f"eye_closure_seconds: {features.current_eye_closure_seconds:.2f}",
        f"jawOpen: {fmt(features.mouth_open_score)}",
        f"yawn_active: {features.yawn_active}",
        f"yaw: {fmt(features.yaw, 1)}",
        f"pitch: {fmt(features.pitch, 1)}",
        f"roll: {fmt(features.roll, 1)}",
        f"head_down: {features.head_down}",
        f"p_drowsy_raw: {_format_yolo_value(features.p_drowsy_raw, features)}",
        f"p_drowsy_ema: {_format_yolo_value(features.p_drowsy_ema, features)}",
        f"p_drowsy_mean_5s: {fmt(metrics.p_drowsy_mean_5s)}",
        f"p_drowsy_high_ratio_5s: {fmt(metrics.p_drowsy_high_ratio_5s)}",
        f"perclos_30s: {fmt(metrics.perclos_30s)}",
        f"risk_score: {fmt(metrics.risk_score)}",
        f"yolo_result_stale: {features.yolo_result_stale}",
        f"yolo_latency_ms: {_format_yolo_value(features.yolo_latency_ms, features, digits=1)}",
        f"camera FPS: {camera_fps:.1f}",
        f"YOLO FPS: {yolo_fps:.1f}",
    ]
    if debug and not features.face_valid and not features.yolo_result_stale:
        lines.append("YOLO display guard: face missing, forcing N/A")
    if calibration_text:
        lines.append(calibration_text)

    for idx, line in enumerate(lines):
        y = 30 + idx * 24
        cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    _draw_yolo_badge(output, badge)
    return output
