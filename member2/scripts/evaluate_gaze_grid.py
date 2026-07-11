from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.human_sensing import (  # noqa: E402
    CameraConfig,
    GazeEstimator,
    create_camera_source,
    default_calibration_targets,
    estimate_head_pose,
    extract_face_landmarks,
    load_calibration_profile,
)
from modules.human_sensing.utils import canonicalize_grid_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated 3x3 gaze-grid accuracy.")
    parser.add_argument("--camera-source", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--calibration-profile", type=Path, required=True)
    parser.add_argument("--mediapipe-model", type=Path)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--dwell-sec", type=float, default=2.0)
    parser.add_argument("--prepare-sec", type=float, default=5.0)
    parser.add_argument("--warmup-sec", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def split_grid(grid: str) -> tuple[str | None, str | None]:
    if grid == "unknown" or "_" not in grid:
        return (None, None)
    row, col = canonicalize_grid_name(grid).split("_", maxsplit=1)
    return (col, row)


def render_target_canvas(width: int, height: int, target, index: int, total: int, phase: str):
    cv2 = importlib.import_module("cv2")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    center = (int(target.screen_x * width), int(target.screen_y * height))
    color = (0, 180, 255) if phase == "warmup" else (0, 0, 255)

    text_y = height - 118 if target.screen_y < 0.32 else 48
    cv2.putText(
        canvas,
        f"Evaluation target: {target.grid} ({index}/{total})",
        (40, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Keep looking at the dot. Press q to abort.",
        (40, text_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    if phase == "warmup":
        cv2.putText(
            canvas,
            "Warmup",
            (40, text_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Recording",
            (40, text_y + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
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
        "Get ready for gaze evaluation",
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


def summarize_records(records: list[dict]) -> dict:
    valid_records = [record for record in records if record["predicted_grid"] != "unknown"]
    correct_records = [record for record in records if record["is_correct"]]

    by_target: dict[str, list[dict]] = defaultdict(list)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        target = record["target_grid"]
        predicted = record["predicted_grid"]
        by_target[target].append(record)
        confusion[target][predicted] += 1

    per_target = {}
    majority_correct = 0
    for target, items in by_target.items():
        predictions = Counter(item["predicted_grid"] for item in items)
        majority_prediction = predictions.most_common(1)[0][0]
        target_confidences = [item["confidence"] for item in items if item["predicted_grid"] != "unknown"]
        accuracy = sum(1 for item in items if item["is_correct"]) / max(1, len(items))
        per_target[target] = {
            "frames": len(items),
            "accuracy": round(accuracy, 4),
            "majority_prediction": majority_prediction,
            "majority_correct": majority_prediction == target,
            "mean_confidence": round(statistics.fmean(target_confidences), 4) if target_confidences else 0.0,
            "predictions": dict(predictions),
        }
        if majority_prediction == target:
            majority_correct += 1

    horizontal_total = 0
    horizontal_correct = 0
    vertical_total = 0
    vertical_correct = 0
    for record in records:
        target_col, target_row = split_grid(record["target_grid"])
        predicted_col, predicted_row = split_grid(record["predicted_grid"])
        if target_col is not None and predicted_col is not None:
            horizontal_total += 1
            horizontal_correct += int(target_col == predicted_col)
        if target_row is not None and predicted_row is not None:
            vertical_total += 1
            vertical_correct += int(target_row == predicted_row)

    return {
        "frame_count": len(records),
        "valid_frame_count": len(valid_records),
        "face_detected_rate": round(len(valid_records) / max(1, len(records)), 4),
        "frame_accuracy": round(len(correct_records) / max(1, len(records)), 4),
        "valid_frame_accuracy": round(len(correct_records) / max(1, len(valid_records)), 4) if valid_records else 0.0,
        "majority_grid_accuracy": round(majority_correct / max(1, len(by_target)), 4),
        "horizontal_accuracy": round(horizontal_correct / max(1, horizontal_total), 4),
        "vertical_accuracy": round(vertical_correct / max(1, vertical_total), 4),
        "per_target": per_target,
        "confusion": {target: dict(counter) for target, counter in confusion.items()},
    }


def main() -> int:
    args = parse_args()
    if args.mediapipe_model:
        os.environ["MEDIAPIPE_FACE_LANDMARKER_MODEL"] = str(args.mediapipe_model.expanduser().resolve())

    cv2 = importlib.import_module("cv2")
    profile = load_calibration_profile(args.calibration_profile)
    config = CameraConfig(
        source_type=args.camera_source,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_depth=args.camera_source == "realsense",
        realsense_serial=args.realsense_serial,
    )
    estimator = GazeEstimator()
    targets = default_calibration_targets()
    records: list[dict] = []

    if not args.no_preview:
        cv2.namedWindow("Evaluation Target", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Evaluation Target", args.width, args.height)
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    try:
        with create_camera_source(config) as camera:
            if not args.no_preview:
                prepare_start = time.time()
                while time.time() - prepare_start < args.prepare_sec:
                    packet = camera.read()
                    frame = packet.color_frame
                    remaining = args.prepare_sec - (time.time() - prepare_start)
                    cv2.imshow("Evaluation Target", render_prepare_canvas(args.width, args.height, remaining))
                    cv2.imshow("Camera Preview", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        return 1

            for index, target in enumerate(targets, start=1):
                start_time = time.time()
                while time.time() - start_time < args.warmup_sec + args.dwell_sec:
                    packet = camera.read()
                    frame = packet.color_frame
                    elapsed = time.time() - start_time
                    phase = "warmup" if elapsed < args.warmup_sec else "record"

                    landmarks = extract_face_landmarks(frame)
                    head_pose = estimate_head_pose(landmarks)
                    prediction = estimator.predict(
                        frame,
                        calibration_profile=profile,
                        slide_id=None,
                        face_landmarks=landmarks,
                        head_pose=head_pose,
                    )

                    if phase == "record":
                        records.append(
                            {
                                "timestamp": prediction.timestamp,
                                "target_grid": target.grid,
                                "predicted_grid": prediction.gaze_grid,
                                "is_correct": prediction.gaze_grid == target.grid,
                                "confidence": prediction.confidence,
                                "stable_duration_sec": prediction.stable_duration_sec,
                                "face_detected": landmarks.face_detected,
                                "head_pose": head_pose.to_dict(),
                            }
                        )

                    if not args.no_preview:
                        target_canvas = render_target_canvas(args.width, args.height, target, index, len(targets), phase)
                        cv2.imshow("Evaluation Target", target_canvas)
                        cv2.imshow("Camera Preview", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            return 1
    finally:
        if not args.no_preview:
            cv2.destroyAllWindows()

    summary = summarize_records(records)
    payload = {
        "created_at": time.time(),
        "calibration_profile": str(args.calibration_profile),
        "camera_source": args.camera_source,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "dwell_sec": args.dwell_sec,
        "prepare_sec": args.prepare_sec,
        "warmup_sec": args.warmup_sec,
        "summary": summary,
        "records": records,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"gaze_eval_{int(payload['created_at'])}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
