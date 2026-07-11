from __future__ import annotations

import importlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AOI, AOIPrediction, CalibrationProfile, FaceLandmarks, GazeFeatures, GazePrediction, HeadPose, Landmark2D
from .utils import (
    GRID_ORDER,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    build_named_landmarks,
    canonicalize_grid_name,
    clamp,
    eye_aspect_ratio,
    euclidean_distance,
    grid_cell_bbox,
    gaze_feature_weight,
    normalized_eye_offsets,
)

_DEFAULT_EXTRACTOR: "FaceLandmarkExtractor | None" = None
_DEFAULT_ESTIMATOR: "GazeEstimator | None" = None


@dataclass(slots=True)
class GazeEstimatorConfig:
    horizontal_threshold: float = 0.16
    vertical_threshold: float = 0.14
    yaw_weight: float = 0.015
    pitch_weight: float = 0.015


@dataclass(slots=True)
class GazeStabilityTracker:
    last_grid: str | None = None
    last_changed_at: float = 0.0

    def stable_duration(self, grid: str, now: float) -> float:
        if grid != self.last_grid:
            self.last_grid = grid
            self.last_changed_at = now
            return 0.0
        return max(0.0, now - self.last_changed_at)


class FaceLandmarkExtractor:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        self._cv2 = importlib.import_module("cv2")
        self._mp = importlib.import_module("mediapipe")
        self._backend = ""
        self._face_mesh = None
        self._face_landmarker = None
        self._last_video_timestamp_ms = 0

        if hasattr(self._mp, "solutions"):
            self._backend = "solutions"
            self._face_mesh = self._mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        if hasattr(self._mp, "tasks"):
            self._backend = "tasks"
            resolved_model_path = resolve_mediapipe_model_path(model_path)
            if resolved_model_path is None:
                raise RuntimeError(
                    "The installed mediapipe package exposes only the Tasks API, but no face landmarker model was found. "
                    "Download `face_landmarker.task` and pass `--mediapipe-model /absolute/path/to/face_landmarker.task`, "
                    "or set `MEDIAPIPE_FACE_LANDMARKER_MODEL`."
                )
            base_options = self._mp.tasks.BaseOptions(model_asset_path=resolved_model_path)
            vision = self._mp.tasks.vision
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
            return

        raise RuntimeError("Unsupported mediapipe installation: neither `solutions` nor `tasks` API is available.")

    def extract(self, frame: np.ndarray) -> FaceLandmarks:
        height, width = frame.shape[:2]
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._face_mesh.process(rgb_frame)
            raw_faces = result.multi_face_landmarks
            if not raw_faces:
                return FaceLandmarks.empty(image_size=(width, height))
            face = raw_faces[0].landmark
        elif self._backend == "tasks":
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = max(int(time.time() * 1000), self._last_video_timestamp_ms + 1)
            self._last_video_timestamp_ms = timestamp_ms
            result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
            raw_faces = getattr(result, "face_landmarks", None)
            if not raw_faces:
                return FaceLandmarks.empty(image_size=(width, height))
            face = raw_faces[0]
        else:
            raise RuntimeError(f"Unsupported FaceLandmarkExtractor backend: {self._backend}")

        if not face:
            return FaceLandmarks.empty(image_size=(width, height))
        points = {
            index: Landmark2D(
                x=landmark.x * width,
                y=landmark.y * height,
                z=landmark.z * width,
            )
            for index, landmark in enumerate(face)
        }
        landmarks = FaceLandmarks(
            points=points,
            named_points={},
            image_size=(width, height),
            face_detected=True,
            iris_available=all(index in points for index in range(468, 478)),
        )
        landmarks.named_points = build_named_landmarks(landmarks)
        return landmarks

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._face_landmarker is not None:
            self._face_landmarker.close()


def resolve_mediapipe_model_path(model_path: str | None = None) -> str | None:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env_model = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if env_model:
        candidates.append(Path(env_model).expanduser())
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "models" / "face_landmarker.task",
            project_root / "data" / "models" / "face_landmarker_v2.task",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def get_default_extractor() -> FaceLandmarkExtractor:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = FaceLandmarkExtractor()
    return _DEFAULT_EXTRACTOR


def extract_face_landmarks(frame: np.ndarray) -> FaceLandmarks:
    return get_default_extractor().extract(frame)


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


def normalize_angle_degrees(value: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return ((value + 180.0) % 360.0) - 180.0


def normalize_head_pose_angles(yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
    yaw = normalize_angle_degrees(yaw)
    pitch = normalize_angle_degrees(pitch)
    roll = normalize_angle_degrees(roll)

    # solvePnP can return a visually front-facing pose with pitch close to +/-180
    # because the generic 3D face model and image coordinate system differ by a
    # near-180-degree x-axis flip. Fold that equivalent representation back near 0.
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0

    return (yaw, pitch, roll)


def estimate_head_pose(face_landmarks: FaceLandmarks) -> HeadPose:
    if not face_landmarks.face_detected:
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0)

    named = face_landmarks.named_points or build_named_landmarks(face_landmarks)
    required = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right"]
    if any(name not in named for name in required):
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0)

    cv2 = importlib.import_module("cv2")
    image_points = np.array(
        [named[name].as_tuple() for name in required],
        dtype="double",
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype="double",
    )
    width, height = face_landmarks.image_size
    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw, pitch, roll = rotation_matrix_to_euler(rotation_matrix)
    yaw, pitch, roll = normalize_head_pose_angles(yaw, pitch, roll)
    projected_points, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    reprojection_error = float(np.mean(np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1)))
    confidence = clamp(1.0 - reprojection_error / max(width, height), 0.0, 1.0)
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        confidence=confidence,
        translation=tuple(float(value) for value in translation_vector.reshape(-1)[:3]),
    )


def extract_gaze_features(
    frame: np.ndarray,
    face_landmarks: FaceLandmarks | None = None,
    head_pose: HeadPose | None = None,
) -> GazeFeatures:
    now = time.time()
    if face_landmarks is None:
        face_landmarks = extract_face_landmarks(frame)
    if head_pose is None:
        head_pose = estimate_head_pose(face_landmarks)
    if not face_landmarks.face_detected:
        return GazeFeatures(timestamp=now, face_detected=False)

    left_x, left_y, right_x, right_y = normalized_eye_offsets(face_landmarks.points)
    left_ear = eye_aspect_ratio(face_landmarks.points, LEFT_EYE_INDICES)
    right_ear = eye_aspect_ratio(face_landmarks.points, RIGHT_EYE_INDICES)
    return GazeFeatures(
        timestamp=now,
        yaw=head_pose.yaw,
        pitch=head_pose.pitch,
        roll=head_pose.roll,
        left_iris_offset_x=left_x,
        left_iris_offset_y=left_y,
        right_iris_offset_x=right_x,
        right_iris_offset_y=right_y,
        left_eye_aspect_ratio=left_ear,
        right_eye_aspect_ratio=right_ear,
        face_detected=True,
    )


class GazeEstimator:
    def __init__(self, config: GazeEstimatorConfig | None = None):
        self.config = config or GazeEstimatorConfig()
        self.tracker = GazeStabilityTracker()

    def _predict_with_calibration(self, features: GazeFeatures, calibration_profile: CalibrationProfile) -> tuple[str, float]:
        feature_vector = features.as_vector(calibration_profile.feature_names)
        metadata = calibration_profile.metadata or {}
        normalization = metadata.get("feature_normalization", {})
        std_values = normalization.get("std")
        if isinstance(std_values, list) and len(std_values) == len(calibration_profile.feature_names):
            scale = np.asarray(std_values, dtype=float)
            scale = np.where(scale < 1e-6, 1.0, scale)
        else:
            scale = np.ones_like(feature_vector)

        raw_weights = metadata.get("feature_weights", {})
        if isinstance(raw_weights, dict):
            weights = np.asarray(
                [float(raw_weights.get(name, gaze_feature_weight(name))) for name in calibration_profile.feature_names],
                dtype=float,
            )
        else:
            weights = np.asarray([gaze_feature_weight(name) for name in calibration_profile.feature_names], dtype=float)

        distances: dict[str, float] = {}
        for grid, centroid in calibration_profile.grid_centroids.items():
            center = np.asarray(centroid, dtype=float)
            normalized_delta = ((feature_vector - center) / scale) * weights
            distances[canonicalize_grid_name(grid)] = float(np.linalg.norm(normalized_delta))
        best_grid = min(distances, key=distances.get)
        sorted_distances = sorted(distances.values())
        best_distance = sorted_distances[0]
        next_distance = sorted_distances[1] if len(sorted_distances) > 1 else best_distance + 1.0
        margin = max(1e-6, next_distance - best_distance)
        confidence = clamp(0.55 + margin / max(next_distance, 1e-6), 0.0, 0.99)
        return (best_grid, confidence)

    def _predict_heuristically(self, features: GazeFeatures) -> tuple[str, float]:
        horizontal_score = ((features.left_iris_offset_x + features.right_iris_offset_x) / 2.0) + (features.yaw * self.config.yaw_weight)
        vertical_score = ((features.left_iris_offset_y + features.right_iris_offset_y) / 2.0) + (features.pitch * self.config.pitch_weight)

        if horizontal_score <= -self.config.horizontal_threshold:
            horizontal = "left"
        elif horizontal_score >= self.config.horizontal_threshold:
            horizontal = "right"
        else:
            horizontal = "center"

        if vertical_score <= -self.config.vertical_threshold:
            vertical = "top"
        elif vertical_score >= self.config.vertical_threshold:
            vertical = "bottom"
        else:
            vertical = "middle"

        confidence = clamp(
            0.45
            + min(abs(horizontal_score) / max(self.config.horizontal_threshold, 1e-6), 1.0) * 0.25
            + min(abs(vertical_score) / max(self.config.vertical_threshold, 1e-6), 1.0) * 0.25,
            0.0,
            0.95,
        )
        return (f"{vertical}_{horizontal}", confidence)

    def predict(
        self,
        frame: np.ndarray,
        calibration_profile: CalibrationProfile | None = None,
        slide_id: int | None = None,
        face_landmarks: FaceLandmarks | None = None,
        head_pose: HeadPose | None = None,
    ) -> GazePrediction:
        features = extract_gaze_features(frame, face_landmarks=face_landmarks, head_pose=head_pose)
        if not features.face_detected:
            return GazePrediction(
                timestamp=features.timestamp,
                slide_id=slide_id,
                gaze_grid="unknown",
                confidence=0.0,
                stable_duration_sec=0.0,
                features=features.as_named_vector(),
                evidence=["face_not_detected"],
            )

        if calibration_profile is not None:
            gaze_grid, confidence = self._predict_with_calibration(features, calibration_profile)
            evidence = ["calibration_profile_applied"]
        else:
            gaze_grid, confidence = self._predict_heuristically(features)
            evidence = ["heuristic_gaze_without_calibration"]

        now = features.timestamp
        stable_duration = self.tracker.stable_duration(gaze_grid, now)
        return GazePrediction(
            timestamp=now,
            slide_id=slide_id,
            gaze_grid=gaze_grid,
            confidence=confidence,
            stable_duration_sec=stable_duration,
            features=features.as_named_vector(),
            evidence=evidence,
        )


def get_default_estimator() -> GazeEstimator:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = GazeEstimator()
    return _DEFAULT_ESTIMATOR


def predict_gaze_grid(
    frame: np.ndarray,
    calibration_profile: CalibrationProfile | None = None,
    slide_id: int | None = None,
) -> GazePrediction:
    return get_default_estimator().predict(frame, calibration_profile=calibration_profile, slide_id=slide_id)


def map_gaze_to_aoi(gaze_prediction: GazePrediction, aois: list[AOI | dict[str, Any]]) -> AOIPrediction:
    normalized_aois = [item if isinstance(item, AOI) else AOI.from_dict(item) for item in aois]
    candidate_scores: dict[str, float] = {}
    evidence = list(gaze_prediction.evidence)

    if gaze_prediction.gaze_grid == "unknown" or not normalized_aois:
        return AOIPrediction(
            timestamp=gaze_prediction.timestamp,
            slide_id=gaze_prediction.slide_id,
            gaze_grid=gaze_prediction.gaze_grid,
            predicted_aoi_id=None,
            confidence=0.0,
            stable_duration_sec=gaze_prediction.stable_duration_sec,
            candidate_scores={},
            evidence=evidence + ["aoi_mapping_skipped"],
        )

    cell = grid_cell_bbox(gaze_prediction.gaze_grid)
    cell_center = cell.center()
    for aoi in normalized_aois:
        overlap = cell.intersection_area(aoi.bbox)
        overlap_ratio = overlap / max(aoi.bbox.area(), 1e-6)
        aoi_center = aoi.bbox.center()
        center_distance = euclidean_distance(cell_center, aoi_center)
        proximity = clamp(1.0 - center_distance / math.sqrt(2.0), 0.0, 1.0)
        score = 0.7 * overlap_ratio + 0.3 * proximity
        candidate_scores[aoi.aoi_id] = round(score, 6)

    best_aoi_id = max(candidate_scores, key=candidate_scores.get)
    final_confidence = clamp(gaze_prediction.confidence * candidate_scores[best_aoi_id], 0.0, 0.99)
    evidence.append(f"best_overlap_aoi={best_aoi_id}")
    return AOIPrediction(
        timestamp=gaze_prediction.timestamp,
        slide_id=gaze_prediction.slide_id,
        gaze_grid=gaze_prediction.gaze_grid,
        predicted_aoi_id=best_aoi_id,
        confidence=final_confidence,
        stable_duration_sec=gaze_prediction.stable_duration_sec,
        candidate_scores=candidate_scores,
        evidence=evidence,
    )
