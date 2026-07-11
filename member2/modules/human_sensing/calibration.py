from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import CalibrationProfile, CalibrationSample, GazeFeatures
from .utils import GRID_ORDER, canonicalize_grid_name, gaze_feature_weight


@dataclass(slots=True)
class CalibrationTarget:
    grid: str
    screen_x: float
    screen_y: float
    dwell_seconds: float = 1.75


def default_calibration_targets() -> list[CalibrationTarget]:
    x_positions = [0.17, 0.50, 0.83]
    y_positions = [0.17, 0.50, 0.83]
    targets: list[CalibrationTarget] = []
    for row_name, y_value in zip(["top", "middle", "bottom"], y_positions, strict=True):
        for col_name, x_value in zip(["left", "center", "right"], x_positions, strict=True):
            targets.append(CalibrationTarget(grid=f"{row_name}_{col_name}", screen_x=x_value, screen_y=y_value))
    return targets


class CalibrationSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.samples: list[CalibrationSample] = []
        self._feature_names: list[str] | None = None

    def record_sample(self, grid: str, features: GazeFeatures, timestamp: float | None = None) -> None:
        if not features.face_detected:
            return
        feature_map = features.as_named_vector()
        if self._feature_names is None:
            self._feature_names = list(feature_map.keys())
        self.samples.append(
            CalibrationSample(
                grid=canonicalize_grid_name(grid),
                feature_names=list(self._feature_names),
                feature_vector=[feature_map[name] for name in self._feature_names],
                timestamp=time.time() if timestamp is None else timestamp,
            )
        )

    def build_profile(self, metadata: dict[str, object] | None = None) -> CalibrationProfile:
        if not self.samples:
            raise ValueError("No calibration samples have been collected")
        if self._feature_names is None:
            raise ValueError("Calibration feature names are missing")

        grouped: dict[str, list[np.ndarray]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.grid, []).append(np.asarray(sample.feature_vector, dtype=float))

        missing = [grid for grid in GRID_ORDER if grid not in grouped]
        if missing:
            raise ValueError(f"Calibration is incomplete. Missing grids: {missing}")

        all_vectors = np.vstack([vector for vectors in grouped.values() for vector in vectors])
        normalization_mean = np.mean(all_vectors, axis=0)
        normalization_std = np.std(all_vectors, axis=0)
        normalization_std = np.where(normalization_std < 1e-6, 1.0, normalization_std)
        feature_weights = {name: gaze_feature_weight(name) for name in self._feature_names}

        centroids = {grid: np.median(vectors, axis=0).round(6).tolist() for grid, vectors in grouped.items()}
        spreads = {grid: np.std(vectors, axis=0).round(6).tolist() for grid, vectors in grouped.items()}
        metadata_payload = dict(metadata or {})
        metadata_payload.update(
            {
                "classifier": "standardized_weighted_centroid",
                "centroid_method": "median",
                "feature_normalization": {
                    "mean": normalization_mean.round(6).tolist(),
                    "std": normalization_std.round(6).tolist(),
                },
                "feature_weights": feature_weights,
                "sample_counts": {grid: len(vectors) for grid, vectors in grouped.items()},
            }
        )
        return CalibrationProfile(
            user_id=self.user_id,
            feature_names=list(self._feature_names),
            grid_centroids=centroids,
            grid_spreads=spreads,
            created_at=time.time(),
            metadata=metadata_payload,
        )


class CalibrationProfileStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe_user_id = user_id.replace("/", "_").replace(" ", "_")
        return self.base_dir / f"{safe_user_id}.json"

    def save(self, profile: CalibrationProfile, path: str | Path | None = None) -> Path:
        output_path = Path(path) if path else self.path_for(profile.user_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profile.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path

    def load(self, path_or_user_id: str | Path) -> CalibrationProfile:
        path = Path(path_or_user_id)
        if not path.exists():
            path = self.path_for(str(path_or_user_id))
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationProfile.from_dict(payload)


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationProfile.from_dict(payload)
