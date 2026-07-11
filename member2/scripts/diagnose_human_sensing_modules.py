from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing import (  # noqa: E402
    CameraConfig,
    FaceStateDetector,
    GazeEstimator,
    HumanSensingHistory,
    create_camera_source,
    estimate_head_pose,
    extract_face_landmarks,
    extract_gaze_features,
    load_calibration_profile,
)
from modules.human_sensing.gaze_estimator import (  # noqa: E402
    normalize_head_pose_angles,
    rotation_matrix_to_euler,
)
from modules.human_sensing.utils import GRID_ORDER, build_named_landmarks, canonicalize_grid_name  # noqa: E402


CAMERA_MODULES = {"camera", "landmarks", "head_pose", "gaze", "face_state", "all"}


def add_common_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-source", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument("--mediapipe-model", type=Path)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--preview", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small diagnostics for individual Human Sensing modules.")
    parser.add_argument(
        "--module",
        choices=["camera", "landmarks", "head_pose", "gaze", "face_state", "calibration_profile", "all"],
        required=True,
    )
    add_common_camera_args(parser)
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def print_payload(name: str, payload: dict[str, Any]) -> None:
    print(f"[{name}] {json.dumps(to_jsonable(payload), ensure_ascii=False)}")


def set_mediapipe_model(path: Path | None) -> None:
    if path:
        os.environ["MEDIAPIPE_FACE_LANDMARKER_MODEL"] = str(path.expanduser().resolve())


def camera_config_from_args(args: argparse.Namespace) -> CameraConfig:
    return CameraConfig(
        source_type=args.camera_source,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_depth=args.camera_source == "realsense",
        realsense_serial=args.realsense_serial,
    )


def frame_stats(frame: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "mean_bgr": [round(float(value), 2) for value in frame.mean(axis=(0, 1))],
        "std_bgr": [round(float(value), 2) for value in frame.std(axis=(0, 1))],
    }


def landmark_summary(landmarks) -> dict[str, Any]:
    named = landmarks.named_points or build_named_landmarks(landmarks)
    keys = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right", "left_iris", "right_iris"]
    named_points = {}
    for key in keys:
        point = named.get(key)
        if point is not None:
            named_points[key] = [round(point.x, 2), round(point.y, 2), round(point.z, 2)]
    return {
        "face_detected": landmarks.face_detected,
        "image_size": list(landmarks.image_size),
        "point_count": len(landmarks.points),
        "iris_available": landmarks.iris_available,
        "named_points": named_points,
    }


def estimate_head_pose_diagnostics(face_landmarks) -> dict[str, Any]:
    if not face_landmarks.face_detected:
        return {"face_detected": False, "success": False}

    named = face_landmarks.named_points or build_named_landmarks(face_landmarks)
    required = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right"]
    missing = [name for name in required if name not in named]
    if missing:
        return {"face_detected": True, "success": False, "missing": missing}

    cv2 = importlib.import_module("cv2")
    image_points = np.array([named[name].as_tuple() for name in required], dtype="double")
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
        return {"face_detected": True, "success": False}

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    raw_yaw, raw_pitch, raw_roll = rotation_matrix_to_euler(rotation_matrix)
    yaw, pitch, roll = normalize_head_pose_angles(raw_yaw, raw_pitch, raw_roll)
    official_pose = estimate_head_pose(face_landmarks)
    projected_points, _ = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    reprojection_error = float(np.mean(np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1)))
    return {
        "face_detected": True,
        "success": True,
        "raw_euler_yaw_pitch_roll": [round(raw_yaw, 3), round(raw_pitch, 3), round(raw_roll, 3)],
        "normalized_yaw_pitch_roll": [round(yaw, 3), round(pitch, 3), round(roll, 3)],
        "official_yaw_pitch_roll": [round(official_pose.yaw, 3), round(official_pose.pitch, 3), round(official_pose.roll, 3)],
        "reprojection_error_px": round(reprojection_error, 3),
        "confidence": round(official_pose.confidence, 4),
        "translation": [round(float(value), 3) for value in translation_vector.reshape(-1)[:3]],
    }


def draw_preview(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    display = frame.copy()
    y = 26
    for line in lines:
        cv2.putText(display, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return display


def inspect_calibration_profile(path: Path) -> int:
    if not path.exists():
        user_id = path.stem
        print_payload(
            "calibration_profile",
            {
                "path": str(path),
                "exists": False,
                "error": "calibration profile not found",
                "next_step": (
                    "python scripts\\run_calibration.py "
                    f"--user-id {user_id} "
                    "--camera-source opencv "
                    "--device-index 0 "
                    "--mediapipe-model data\\models\\face_landmarker.task "
                    "--dwell-sec 2.0"
                ),
            },
        )
        return 1

    profile = load_calibration_profile(path)
    raw_grids = list(profile.grid_centroids.keys())
    canonical_grids = [canonicalize_grid_name(grid) for grid in raw_grids]
    missing = [grid for grid in GRID_ORDER if grid not in canonical_grids]
    legacy = {grid: canonicalize_grid_name(grid) for grid in raw_grids if grid != canonicalize_grid_name(grid)}
    payload = {
        "path": str(path),
        "user_id": profile.user_id,
        "feature_names": profile.feature_names,
        "raw_grid_keys": raw_grids,
        "canonical_grid_keys": canonical_grids,
        "missing_canonical_grids": missing,
        "legacy_key_mapping": legacy,
        "grid_count": len(raw_grids),
        "is_complete": not missing,
        "metadata": profile.metadata,
    }
    print_payload("calibration_profile", payload)
    return 0 if not missing else 1


def run_camera_diagnostics(args: argparse.Namespace) -> int:
    cv2 = importlib.import_module("cv2")
    profile = load_calibration_profile(args.calibration_profile) if args.calibration_profile else None
    estimator = GazeEstimator()
    face_state_detector = FaceStateDetector()
    history = HumanSensingHistory()
    config = camera_config_from_args(args)

    if args.preview:
        cv2.namedWindow("Human Sensing Module Diagnostic", cv2.WINDOW_NORMAL)

    try:
        with create_camera_source(config) as camera:
            frame_index = 0
            while True:
                packet = camera.read()
                frame = packet.color_frame
                payload: dict[str, Any] = {
                    "frame_index": packet.frame_index,
                    "timestamp": packet.timestamp,
                    "source": packet.source,
                }
                preview_lines = [f"module: {args.module}", f"frame: {packet.frame_index}"]

                if args.module in {"camera", "all"}:
                    payload["camera"] = frame_stats(frame)
                    preview_lines.append(f"shape: {frame.shape}")

                landmarks = None
                if args.module in {"landmarks", "head_pose", "gaze", "face_state", "all"}:
                    landmarks = extract_face_landmarks(frame)
                    payload["landmarks"] = landmark_summary(landmarks)
                    preview_lines.append(f"face: {landmarks.face_detected}")

                head_pose = None
                if args.module in {"head_pose", "gaze", "face_state", "all"} and landmarks is not None:
                    head_pose = estimate_head_pose(landmarks)
                    payload["head_pose"] = estimate_head_pose_diagnostics(landmarks)
                    preview_lines.append(f"yaw/pitch/roll: {head_pose.yaw:.1f}/{head_pose.pitch:.1f}/{head_pose.roll:.1f}")

                if args.module in {"gaze", "all"} and landmarks is not None and head_pose is not None:
                    features = extract_gaze_features(frame, face_landmarks=landmarks, head_pose=head_pose)
                    prediction = estimator.predict(
                        frame,
                        calibration_profile=profile,
                        face_landmarks=landmarks,
                        head_pose=head_pose,
                    )
                    payload["gaze"] = {
                        "prediction": prediction.to_dict(),
                        "features": features.as_named_vector(),
                        "calibration_applied": profile is not None,
                    }
                    preview_lines.append(f"grid: {prediction.gaze_grid}")
                    preview_lines.append(f"gaze_conf: {prediction.confidence:.2f}")

                if args.module in {"face_state", "all"} and landmarks is not None and head_pose is not None:
                    face_state = face_state_detector.detect_face_state_signals(
                        face_landmarks=landmarks,
                        history=history,
                        head_pose=head_pose,
                    )
                    payload["face_state"] = face_state.to_dict()
                    preview_lines.append(f"screen_facing: {face_state.screen_facing_score:.2f}")
                    preview_lines.append(f"MAR/EAR: {face_state.mouth_aspect_ratio:.2f}/{face_state.eye_aspect_ratio:.2f}")

                if frame_index % max(1, args.print_every) == 0:
                    print_payload(args.module, payload)

                if args.preview:
                    cv2.imshow("Human Sensing Module Diagnostic", draw_preview(frame, preview_lines))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if args.preview:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    print(
        "Note: prefer the dedicated scripts diagnose_camera.py, diagnose_landmarks.py, "
        "diagnose_head_pose.py, diagnose_gaze.py, and diagnose_face_state.py for module-by-module checks."
    )
    set_mediapipe_model(args.mediapipe_model)
    if args.module == "calibration_profile":
        if args.calibration_profile is None:
            raise SystemExit("--calibration-profile is required for --module calibration_profile")
        return inspect_calibration_profile(args.calibration_profile)
    return run_camera_diagnostics(args)


if __name__ == "__main__":
    raise SystemExit(main())
