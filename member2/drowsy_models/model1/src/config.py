from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30, "mirror": True},
    "models": {
        "face_landmarker_path": "models/face_landmarker.task",
        "yolo_local_path": "models/best.pt",
        "yolo_repo_id": "mosesb/drowsiness-detection-yolo-cls",
        "yolo_filename": "best.pt",
        "yolo_imgsz": 224,
        "yolo_device": 0,
        "yolo_half": True,
        "yolo_every_n_frames": 3,
    },
    "mediapipe": {
        "num_faces": 1,
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "face_crop": {
        "horizontal_margin": 0.20,
        "top_margin": 0.20,
        "bottom_margin": 0.25,
        "min_face_width_pixels": 120,
        "min_face_height_pixels": 120,
    },
    "calibration": {
        "enabled": True,
        "duration_seconds": 15,
        "default_eye_closed_threshold": 0.65,
        "default_yawn_threshold": 0.70,
    },
    "temporal": {
        "short_window_seconds": 5,
        "long_window_seconds": 30,
        "yolo_ema_alpha": 0.20,
        "yolo_stale_seconds": 0.50,
        "face_missing_unknown_seconds": 0.70,
    },
    "thresholds": {
        "suspect_yolo_probability": 0.65,
        "drowsy_yolo_probability": 0.75,
        "critical_yolo_probability": 0.90,
        "suspect_perclos": 0.15,
        "drowsy_perclos": 0.25,
        "long_eye_closure_seconds": 0.80,
        "critical_eye_closure_seconds": 1.50,
        "yawn_min_seconds": 1.20,
        "yawn_event_gap_seconds": 2.00,
        "head_down_delta_degrees": 15,
        "head_down_min_seconds": 1.50,
        "recovery_yolo_probability": 0.45,
        "recovery_seconds": 2.00,
    },
    "alert": {"enabled": False, "cooldown_seconds": 3, "enable_audio": False},
    "logging": {
        "enabled": True,
        "output_directory": "logs",
        "save_video": False,
        "save_face_crops": False,
    },
}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    if not path.exists():
        return config
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    _deep_merge(config, loaded)
    return config


def resolve_config_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (Path(project_root) / candidate).resolve()


def resolve_runtime_paths(config: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    base_dir = Path(base_dir).resolve()

    for section, key in (
        ("models", "face_landmarker_path"),
        ("models", "yolo_local_path"),
        ("logging", "output_directory"),
    ):
        value = resolved.get(section, {}).get(key)
        if not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        resolved[section][key] = str(path if path.is_absolute() else (base_dir / path).resolve())

    return resolved


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
