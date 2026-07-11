from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing import (  # noqa: E402
    CameraConfig,
    GazeEstimator,
    HumanSensingHistory,
    create_camera_source,
    detect_learning_state,
    estimate_head_pose,
    extract_face_landmarks,
    load_calibration_profile,
    map_gaze_to_aoi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Member 2 Human Sensing demo pipeline.")
    parser.add_argument("--camera-source", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--slide-id", type=int, default=1)
    parser.add_argument("--aoi-json", type=Path, default=PROJECT_ROOT / "examples" / "sample_aois.json")
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument("--mediapipe-model", type=Path)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--print-every", type=int, default=15)
    return parser.parse_args()


def load_aois(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "aois" in payload:
        return list(payload["aois"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported AOI payload format in {path}")


def draw_overlay(frame, pose, aoi_prediction, learning_state):
    cv2 = importlib.import_module("cv2")
    display = frame.copy()
    height, width = display.shape[:2]
    cv2.line(display, (width // 3, 0), (width // 3, height), (64, 64, 64), 1)
    cv2.line(display, (2 * width // 3, 0), (2 * width // 3, height), (64, 64, 64), 1)
    cv2.line(display, (0, height // 3), (width, height // 3), (64, 64, 64), 1)
    cv2.line(display, (0, 2 * height // 3), (width, 2 * height // 3), (64, 64, 64), 1)

    lines = [
        f"grid: {aoi_prediction.gaze_grid}",
        f"aoi: {getattr(aoi_prediction, 'predicted_aoi_id', None)}",
        f"confidence: {aoi_prediction.confidence:.2f}",
        f"stable_sec: {aoi_prediction.stable_duration_sec:.2f}",
        f"yaw/pitch/roll: {pose.yaw:.1f} / {pose.pitch:.1f} / {pose.roll:.1f}",
        f"screen_facing: {learning_state.screen_facing_score:.2f}",
        f"yawn_count_3min: {learning_state.yawn_count_last_3min}",
        f"eyes_closed: {learning_state.eyes_closed}",
        f"fatigue: {learning_state.fatigue_signal_score:.2f}",
        f"review_needed: {learning_state.possible_review_needed}",
    ]
    y = 24
    for line in lines:
        cv2.putText(display, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return display


def main() -> int:
    args = parse_args()
    if args.mediapipe_model:
        os.environ["MEDIAPIPE_FACE_LANDMARKER_MODEL"] = str(args.mediapipe_model.expanduser().resolve())

    config = CameraConfig(
        source_type=args.camera_source,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_depth=args.camera_source == "realsense",
        realsense_serial=args.realsense_serial,
    )
    profile = load_calibration_profile(args.calibration_profile) if args.calibration_profile else None
    aois = load_aois(args.aoi_json)
    estimator = GazeEstimator()
    history = HumanSensingHistory()
    frame_count = 0

    cv2 = None
    if not args.no_preview:
        cv2 = importlib.import_module("cv2")
        cv2.namedWindow("Human Sensing Demo", cv2.WINDOW_NORMAL)

    try:
        with create_camera_source(config) as camera:
            while True:
                packet = camera.read()
                frame = packet.color_frame
                landmarks = extract_face_landmarks(frame)
                head_pose = estimate_head_pose(landmarks)
                gaze_prediction = estimator.predict(
                    frame,
                    calibration_profile=profile,
                    slide_id=args.slide_id,
                    face_landmarks=landmarks,
                    head_pose=head_pose,
                )
                target_prediction = map_gaze_to_aoi(gaze_prediction, aois) if aois else gaze_prediction
                learning_state = detect_learning_state(
                    frame,
                    landmarks,
                    history,
                    head_pose=head_pose,
                    gaze_prediction=target_prediction,
                )

                if frame_count % max(1, args.print_every) == 0:
                    payload = {
                        "timestamp": target_prediction.timestamp,
                        "gaze_prediction": target_prediction.to_dict(),
                        "learning_state": learning_state.to_dict(),
                    }
                    print(json.dumps(payload, ensure_ascii=False))

                if cv2 is not None:
                    display = draw_overlay(frame, head_pose, target_prediction, learning_state)
                    cv2.imshow("Human Sensing Demo", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_count += 1
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
