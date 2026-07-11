from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing import (  # noqa: E402
    CalibrationProfileStore,
    CalibrationSession,
    CameraConfig,
    create_camera_source,
    default_calibration_targets,
    estimate_head_pose,
    extract_face_landmarks,
    extract_gaze_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect 9-point gaze calibration data.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--camera-source", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dwell-sec", type=float, default=1.75)
    parser.add_argument("--warmup-sec", type=float, default=1.5)
    parser.add_argument("--prepare-sec", type=float, default=5.0)
    parser.add_argument("--mediapipe-model", type=Path)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "calibration_profiles")
    return parser.parse_args()


def render_target_canvas(width: int, height: int, target, index: int, total: int, phase: str):
    cv2 = importlib.import_module("cv2")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    center = (int(target.screen_x * width), int(target.screen_y * height))
    color = (0, 180, 255) if phase == "warmup" else (0, 0, 255)

    text_y = height - 118 if target.screen_y < 0.32 else 48
    cv2.putText(
        canvas,
        f"Look at the red dot: {target.grid} ({index}/{total})",
        (40, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Warmup" if phase == "warmup" else "Recording",
        (40, text_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Press q to abort",
        (40, text_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (center[0] - 72, center[1]), (center[0] + 72, center[1]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (center[0], center[1] - 72), (center[0], center[1] + 72), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 30, color, -1)
    cv2.circle(canvas, center, 58, (255, 255, 255), 3)
    return canvas


def render_prepare_canvas(width: int, height: int, remaining_sec: float):
    cv2 = importlib.import_module("cv2")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Get ready for calibration",
        (40, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Starting in {max(0.0, remaining_sec):.1f}s",
        (40, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Sit still, face the screen, then follow the red dot.",
        (40, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    if args.mediapipe_model:
        os.environ["MEDIAPIPE_FACE_LANDMARKER_MODEL"] = str(args.mediapipe_model.expanduser().resolve())

    cv2 = importlib.import_module("cv2")
    config = CameraConfig(
        source_type=args.camera_source,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_depth=args.camera_source == "realsense",
        realsense_serial=args.realsense_serial,
    )
    session = CalibrationSession(user_id=args.user_id)
    targets = default_calibration_targets()
    cv2.namedWindow("Calibration Target", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Target", args.width, args.height)
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    try:
        with create_camera_source(config) as camera:
            prepare_start = time.time()
            while time.time() - prepare_start < args.prepare_sec:
                packet = camera.read()
                frame = packet.color_frame
                remaining = args.prepare_sec - (time.time() - prepare_start)
                cv2.imshow("Calibration Target", render_prepare_canvas(args.width, args.height, remaining))
                cv2.imshow("Camera Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 1

            for index, target in enumerate(targets, start=1):
                target.dwell_seconds = args.dwell_sec
                start_time = time.time()
                while time.time() - start_time < args.warmup_sec + target.dwell_seconds:
                    packet = camera.read()
                    frame = packet.color_frame
                    elapsed = time.time() - start_time
                    phase = "warmup" if elapsed < args.warmup_sec else "record"

                    landmarks = extract_face_landmarks(frame)
                    head_pose = estimate_head_pose(landmarks)
                    if phase == "record":
                        features = extract_gaze_features(frame, face_landmarks=landmarks, head_pose=head_pose)
                        session.record_sample(target.grid, features)

                    target_canvas = render_target_canvas(args.width, args.height, target, index, len(targets), phase)
                    cv2.imshow("Calibration Target", target_canvas)
                    cv2.imshow("Camera Preview", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        return 1

            profile = session.build_profile(
                metadata={
                    "camera_source": args.camera_source,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "warmup_sec": args.warmup_sec,
                    "dwell_sec": args.dwell_sec,
                }
            )
            store = CalibrationProfileStore(args.output)
            output_path = store.save(profile)
            print(f"Saved calibration profile to: {output_path}")
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
