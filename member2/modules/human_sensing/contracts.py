from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

GridName = Literal[
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


def dataclass_to_dict(value: Any) -> Any:
    """Convert nested dataclasses into plain Python containers."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    return value


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float, float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"bbox must have four values, got {values!r}")
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersection_area(self, other: "BBox") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class AOI:
    aoi_id: str
    bbox: BBox
    aoi_type: str = "unknown"
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AOI":
        return cls(
            aoi_id=str(payload["aoi_id"]),
            bbox=BBox.from_sequence(payload["bbox"]),
            aoi_type=str(payload.get("type", payload.get("aoi_type", "unknown"))),
            text=str(payload.get("text", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "aoi_id": self.aoi_id,
            "bbox": self.bbox.to_list(),
            "type": self.aoi_type,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FramePacket:
    timestamp: float
    color_frame: np.ndarray | None
    depth_frame: np.ndarray | None = None
    frame_index: int = 0
    source: str = "unknown"
    intrinsics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Landmark2D:
    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(slots=True)
class FaceLandmarks:
    points: dict[int, Landmark2D]
    named_points: dict[str, Landmark2D]
    image_size: tuple[int, int]
    face_detected: bool = False
    iris_available: bool = False

    @classmethod
    def empty(cls, image_size: tuple[int, int] = (0, 0)) -> "FaceLandmarks":
        return cls(points={}, named_points={}, image_size=image_size, face_detected=False, iris_available=False)

    def get(self, index: int) -> Landmark2D | None:
        return self.points.get(index)

    def get_named(self, name: str) -> Landmark2D | None:
        return self.named_points.get(name)


@dataclass(slots=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    confidence: float = 0.0
    translation: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class GazeFeatures:
    timestamp: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    left_iris_offset_x: float = 0.0
    left_iris_offset_y: float = 0.0
    right_iris_offset_x: float = 0.0
    right_iris_offset_y: float = 0.0
    left_eye_aspect_ratio: float = 0.0
    right_eye_aspect_ratio: float = 0.0
    face_detected: bool = False

    def as_named_vector(self) -> dict[str, float]:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "left_iris_offset_x": self.left_iris_offset_x,
            "left_iris_offset_y": self.left_iris_offset_y,
            "right_iris_offset_x": self.right_iris_offset_x,
            "right_iris_offset_y": self.right_iris_offset_y,
            "left_eye_aspect_ratio": self.left_eye_aspect_ratio,
            "right_eye_aspect_ratio": self.right_eye_aspect_ratio,
        }

    def as_vector(self, feature_names: list[str] | tuple[str, ...]) -> np.ndarray:
        vector = [self.as_named_vector().get(name, 0.0) for name in feature_names]
        return np.asarray(vector, dtype=float)


@dataclass(slots=True)
class CalibrationSample:
    grid: GridName
    feature_names: list[str]
    feature_vector: list[float]
    timestamp: float


@dataclass(slots=True)
class CalibrationProfile:
    user_id: str
    feature_names: list[str]
    grid_centroids: dict[str, list[float]]
    grid_spreads: dict[str, list[float]]
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationProfile":
        return cls(
            user_id=str(payload["user_id"]),
            feature_names=list(payload["feature_names"]),
            grid_centroids={key: list(values) for key, values in payload["grid_centroids"].items()},
            grid_spreads={key: list(values) for key, values in payload.get("grid_spreads", {}).items()},
            created_at=float(payload["created_at"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class GazePrediction:
    timestamp: float
    slide_id: int | None
    gaze_grid: str
    confidence: float
    stable_duration_sec: float
    features: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class AOIPrediction:
    timestamp: float
    slide_id: int | None
    gaze_grid: str
    predicted_aoi_id: str | None
    confidence: float
    stable_duration_sec: float
    candidate_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class FaceStateSignals:
    timestamp: float
    face_detected: bool
    screen_facing_score: float
    yawn_detected: bool
    yawn_count_last_3min: int
    eyes_closed: bool
    eye_closure_duration_sec: float
    head_down: bool
    mouth_aspect_ratio: float
    eye_aspect_ratio: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class LearningState:
    timestamp: float
    face_detected: bool
    screen_facing_score: float
    yawn_detected: bool
    yawn_count_last_3min: int
    eyes_closed: bool
    eye_closure_duration_sec: float
    head_down: bool
    fatigue_signal_score: float
    possible_review_needed: bool
    stable_attention_sec: float = 0.0
    repeated_attention_to_same_aoi: bool = False
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(slots=True)
class GazeEvent:
    timestamp: float
    gaze_grid: str
    predicted_aoi_id: str | None
    confidence: float
    stable_duration_sec: float


@dataclass(slots=True)
class HumanSensingHistory:
    current_yawn_started_at: float | None = None
    yawn_timestamps: list[float] = field(default_factory=list)
    eye_closed_since: float | None = None
    gaze_events: list[GazeEvent] = field(default_factory=list)

    def record_gaze(self, prediction: GazePrediction | AOIPrediction) -> None:
        predicted_aoi_id = getattr(prediction, "predicted_aoi_id", None)
        self.gaze_events.append(
            GazeEvent(
                timestamp=prediction.timestamp,
                gaze_grid=prediction.gaze_grid,
                predicted_aoi_id=predicted_aoi_id,
                confidence=prediction.confidence,
                stable_duration_sec=prediction.stable_duration_sec,
            )
        )

    def prune(self, now: float, history_window_sec: float = 180.0) -> None:
        cutoff = now - history_window_sec
        self.yawn_timestamps = [ts for ts in self.yawn_timestamps if ts >= cutoff]
        self.gaze_events = [event for event in self.gaze_events if event.timestamp >= cutoff]
