from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing import (  # noqa: E402
    CameraConfig,
    create_camera_source,
    default_calibration_targets,
    estimate_head_pose,
    extract_face_landmarks,
    extract_gaze_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled 3x3 gaze data for later model training.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--camera-source", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mediapipe-model", type=Path)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "gaze_datasets")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prepare-sec", type=float, default=5.0)
    parser.add_argument("--warmup-sec", type=float, default=1.5)
    parser.add_argument("--dwell-sec", type=float, default=2.0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=3, help="Save/record every Nth frame during target dwell.")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--keep-undetected", action="store_true", help="Keep samples even when no face is detected.")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def draw_camera_inset(canvas: np.ndarray, frame: np.ndarray, target) -> None:
    cv2 = importlib.import_module("cv2")
    height, width = canvas.shape[:2]
    inset_w = min(320, max(220, width // 4))
    inset_h = int(inset_w * 9 / 16)
    margin = 28
    y1 = margin if target.screen_y > 0.5 else height - inset_h - margin
    if y1 == margin:
        x1 = width - inset_w - margin
    else:
        x1 = margin if target.screen_x > 0.5 else width - inset_w - margin
    x2 = x1 + inset_w
    y2 = y1 + inset_h

    preview = cv2.resize(frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    canvas[y1:y2, x1:x2] = preview
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(canvas, (x1, y1), (x2, y1 + 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "Camera preview",
        (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def render_target_canvas(width: int, height: int, target, repetition: int, repetitions: int, phase: str, preview_frame=None):
    cv2 = importlib.import_module("cv2")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    center = (int(target.screen_x * width), int(target.screen_y * height))
    color = (0, 180, 255) if phase == "warmup" else (0, 0, 255)

    text_y = height - 118 if target.screen_y < 0.32 else 52
    cv2.putText(
        canvas,
        f"{target.grid} | round {repetition}/{repetitions}",
        (40, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Check camera in warmup, then look at the dot.",
        (40, text_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Warmup" if phase == "warmup" else "Recording",
        (40, text_y + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (center[0] - 72, center[1]), (center[0] + 72, center[1]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (center[0], center[1] - 72), (center[0], center[1] + 72), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 30, color, -1)
    cv2.circle(canvas, center, 58, (255, 255, 255), 3)
    if phase == "warmup" and preview_frame is not None:
        draw_camera_inset(canvas, preview_frame, target)
    return canvas


def render_prepare_canvas(width: int, height: int, remaining_sec: float):
    cv2 = importlib.import_module("cv2")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Get ready for gaze data collection",
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


def create_dataset_dir(args: argparse.Namespace) -> Path:
    dataset_id = f"{args.user_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    safe_dataset_id = dataset_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    dataset_dir = args.output / safe_dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=False)
    if args.save_frames:
        (dataset_dir / "frames").mkdir(parents=True, exist_ok=True)
    return dataset_dir


def main() -> int:
    args = parse_args()
    if args.mediapipe_model:
        os.environ["MEDIAPIPE_FACE_LANDMARKER_MODEL"] = str(args.mediapipe_model.expanduser().resolve())

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    cv2 = importlib.import_module("cv2")
    dataset_dir = create_dataset_dir(args)
    samples_path = dataset_dir / "samples.jsonl"
    metadata_path = dataset_dir / "metadata.json"

    config = CameraConfig(
        source_type=args.camera_source,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_depth=args.camera_source == "realsense",
        realsense_serial=args.realsense_serial,
    )
    targets = default_calibration_targets()
    metadata = {
        "user_id": args.user_id,
        "dataset_dir": str(dataset_dir),
        "created_at": time.time(),
        "camera_source": args.camera_source,
        "device_index": args.device_index,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "repetitions": args.repetitions,
        "prepare_sec": args.prepare_sec,
        "warmup_sec": args.warmup_sec,
        "dwell_sec": args.dwell_sec,
        "target_order": "randomized_each_repetition",
        "save_frames": args.save_frames,
        "frame_stride": args.frame_stride,
        "grid_order": [target.grid for target in targets],
        "sample_format": "jsonl; one labeled frame/features sample per line",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    cv2.namedWindow("Gaze Data Target", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Data Target", args.width, args.height)
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    recorded = 0
    skipped_no_face = 0
    target_counts = {target.grid: 0 for target in targets}

    try:
        with create_camera_source(config) as camera, samples_path.open("w", encoding="utf-8") as samples_file:
            prepare_start = time.time()
            while time.time() - prepare_start < args.prepare_sec:
                packet = camera.read()
                frame = packet.color_frame
                remaining = args.prepare_sec - (time.time() - prepare_start)
                cv2.imshow("Gaze Data Target", render_prepare_canvas(args.width, args.height, remaining))
                cv2.imshow("Camera Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt

            for repetition in range(1, args.repetitions + 1):
                round_targets = list(targets)
                random.shuffle(round_targets)

                for target in round_targets:
                    start_time = time.time()
                    frame_in_target = 0
                    while time.time() - start_time < args.warmup_sec + args.dwell_sec:
                        packet = camera.read()
                        frame = packet.color_frame
                        elapsed = time.time() - start_time
                        phase = "warmup" if elapsed < args.warmup_sec else "record"

                        if phase == "record" and frame_in_target % args.frame_stride == 0:
                            landmarks = extract_face_landmarks(frame)
                            head_pose = estimate_head_pose(landmarks)
                            features = extract_gaze_features(frame, face_landmarks=landmarks, head_pose=head_pose)

                            if landmarks.face_detected or args.keep_undetected:
                                frame_path = None
                                if args.save_frames:
                                    frame_name = f"{recorded:06d}_{target.grid}.jpg"
                                    frame_path = Path("frames") / frame_name
                                    cv2.imwrite(
                                        str(dataset_dir / frame_path),
                                        frame,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                                    )

                                sample = {
                                    "sample_index": recorded,
                                    "timestamp": packet.timestamp,
                                    "frame_index": packet.frame_index,
                                    "target_grid": target.grid,
                                    "target_screen_xy": [target.screen_x, target.screen_y],
                                    "repetition": repetition,
                                    "camera_source": packet.source,
                                    "face_detected": landmarks.face_detected,
                                    "iris_available": landmarks.iris_available,
                                    "image_size": list(landmarks.image_size),
                                    "head_pose": head_pose.to_dict(),
                                    "features": features.as_named_vector(),
                                    "frame_path": str(frame_path).replace("\\", "/") if frame_path else None,
                                }
                                samples_file.write(json.dumps(sample, ensure_ascii=True, default=json_default) + "\n")
                                samples_file.flush()
                                recorded += 1
                                target_counts[target.grid] += 1
                            else:
                                skipped_no_face += 1

                        target_canvas = render_target_canvas(
                            args.width,
                            args.height,
                            target,
                            repetition,
                            args.repetitions,
                            phase,
                            preview_frame=frame if phase == "warmup" else None,
                        )
                        cv2.imshow("Gaze Data Target", target_canvas)
                        cv2.imshow("Camera Preview", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            raise KeyboardInterrupt
                        frame_in_target += 1
    except KeyboardInterrupt:
        print("Collection stopped by user.")
    finally:
        cv2.destroyAllWindows()

    metadata["finished_at"] = time.time()
    metadata["recorded_samples"] = recorded
    metadata["skipped_no_face"] = skipped_no_face
    metadata["target_counts"] = target_counts
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved dataset to: {dataset_dir}")
    print(f"Samples: {recorded}")
    print(f"Skipped no-face frames: {skipped_no_face}")
    print(f"Metadata: {metadata_path}")
    print(f"Samples JSONL: {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
