from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.calibration import CalibrationCollector, CalibrationResult
from src.camera import CameraError, CameraStream
from src.config import load_config, resolve_config_path, resolve_runtime_paths
from src.face_cropper import crop_face_roi
from src.face_landmarker import FaceLandmarkerError, MediaPipeFaceLandmarker
from src.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from src.fusion_engine import FusionEngine
from src.logger import SessionLogger
from src.state_machine import StateMachine, StateMachineThresholds
from src.temporal_buffer import TemporalBuffer
from src.visualizer import draw_overlay
from src.yolo_classifier import resolve_device_config
from src.yolo_worker import YoloWorker

PROJECT_ROOT = Path(__file__).resolve().parent


def _get_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required to run the application") from exc
    return cv2


def _default_calibration_result(config: dict) -> CalibrationResult:
    calibration_cfg = config["calibration"]
    return CalibrationResult(
        enabled=bool(calibration_cfg.get("enabled", True)),
        complete=False,
        quality_ok=False,
        quality_message="校准中",
        eye_closed_threshold=float(calibration_cfg["default_eye_closed_threshold"]),
        yawn_threshold=float(calibration_cfg["default_yawn_threshold"]),
        baseline_pitch=0.0,
        sample_count=0,
        valid_seconds=0.0,
        p_drowsy_samples=[],
    )


def _build_feature_config(config: dict) -> FeatureExtractorConfig:
    return FeatureExtractorConfig(
        yolo_ema_alpha=float(config["temporal"]["yolo_ema_alpha"]),
        yolo_stale_seconds=float(config["temporal"]["yolo_stale_seconds"]),
        yolo_max_age_seconds=2.0,
        head_down_delta_degrees=float(config["thresholds"]["head_down_delta_degrees"]),
        head_down_min_seconds=float(config["thresholds"]["head_down_min_seconds"]),
        max_abs_yaw_for_valid_classification=35.0,
        max_abs_pitch_for_valid_classification=40.0,
        min_face_width_pixels=int(config["face_crop"]["min_face_width_pixels"]),
        min_face_height_pixels=int(config["face_crop"]["min_face_height_pixels"]),
    )


def _update_fps(previous_timestamp: float | None, timestamp: float, current_fps: float) -> float:
    if previous_timestamp is None:
        return current_fps
    dt = max(timestamp - previous_timestamp, 1e-6)
    instant = 1.0 / dt
    if current_fps <= 0:
        return instant
    return 0.85 * current_fps + 0.15 * instant


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time drowsiness signal observer")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, PROJECT_ROOT)
    config = resolve_runtime_paths(load_config(config_path), config_path.parent)

    cv2 = _get_cv2()
    device, half = resolve_device_config(
        requested_device=config["models"].get("yolo_device"),
        requested_half=config["models"].get("yolo_half"),
    )
    print(f"YOLO runtime device={device}, half={half}")

    camera_cfg = config["camera"]
    camera = CameraStream(
        index=int(camera_cfg["index"]),
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        mirror=bool(camera_cfg["mirror"]),
    )

    try:
        camera.open()
    except CameraError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    landmarker = None
    yolo_worker = None
    logger = None
    try:
        landmarker = MediaPipeFaceLandmarker(
            model_path=str(config["models"]["face_landmarker_path"]),
            mediapipe_config=config["mediapipe"],
            face_crop_config=config["face_crop"],
        )

        try:
            yolo_worker = YoloWorker(
                local_path=str(config["models"]["yolo_local_path"]),
                repo_id=str(config["models"]["yolo_repo_id"]),
                filename=str(config["models"]["yolo_filename"]),
                imgsz=int(config["models"]["yolo_imgsz"]),
                device=device,
                half=half,
            )
        except Exception as exc:
            print(f"YOLO disabled, falling back to MediaPipe-only mode: {exc}", file=sys.stderr)
            yolo_worker = None

        extractor = FeatureExtractor(_build_feature_config(config))
        buffer = TemporalBuffer()
        fusion_engine = FusionEngine(
            short_window_seconds=float(config["temporal"]["short_window_seconds"]),
            long_window_seconds=float(config["temporal"]["long_window_seconds"]),
            yawn_min_seconds=float(config["thresholds"]["yawn_min_seconds"]),
        )
        state_machine = StateMachine(StateMachineThresholds.from_mapping({**config["thresholds"], **config["temporal"]}))
        logger = SessionLogger(
            enabled=bool(config["logging"]["enabled"]),
            output_directory=str(config["logging"]["output_directory"]),
        )

        calibration_result = _default_calibration_result(config)
        calibration_enabled = bool(config["calibration"]["enabled"])
        calibration_collector = CalibrationCollector(
            duration_seconds=float(config["calibration"]["duration_seconds"]),
            default_eye_closed_threshold=float(config["calibration"]["default_eye_closed_threshold"]),
            default_yawn_threshold=float(config["calibration"]["default_yawn_threshold"]),
            max_abs_yaw_for_sample=35.0,
            max_abs_pitch_for_sample=40.0,
        )
        calibration_start = time.monotonic()
        calibrating = calibration_enabled

        debug = True
        recording = bool(config["logging"]["enabled"])
        last_camera_frame_ts = None
        last_yolo_frame_ts = None
        camera_fps = 0.0
        yolo_fps = 0.0
        last_yolo_seen_ts = None
        frame_id = 0

        while True:
            ok, frame, timestamp = camera.read()
            camera_fps = _update_fps(last_camera_frame_ts, timestamp, camera_fps)
            last_camera_frame_ts = timestamp

            if not ok or frame is None:
                continue

            face_observation = landmarker.detect(frame, timestamp)
            bbox = face_observation.bbox
            if bbox is not None:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            else:
                width = 0
                height = 0

            if (
                yolo_worker is not None
                and bbox is not None
                and width >= int(config["face_crop"]["min_face_width_pixels"])
                and height >= int(config["face_crop"]["min_face_height_pixels"])
                and frame_id % int(config["models"]["yolo_every_n_frames"]) == 0
            ):
                face_roi = crop_face_roi(frame, bbox, output_size=int(config["models"]["yolo_imgsz"]))
                if face_roi is not None:
                    yolo_worker.submit(frame_id=frame_id, timestamp=timestamp, face_roi=face_roi)

            yolo_result = yolo_worker.get_latest_result() if yolo_worker is not None else None
            if yolo_result is not None and yolo_result.valid:
                if last_yolo_seen_ts is None or yolo_result.timestamp > last_yolo_seen_ts:
                    yolo_fps = _update_fps(last_yolo_frame_ts, yolo_result.timestamp, yolo_fps)
                    last_yolo_frame_ts = yolo_result.timestamp
                    last_yolo_seen_ts = yolo_result.timestamp

            features = extractor.update(face_observation, yolo_result, calibration_result, timestamp)
            buffer.append(features)
            buffer.prune(35.0)
            metrics = fusion_engine.compute_metrics(buffer, timestamp)

            if calibrating:
                calibration_collector.update(features)
                remaining = float(config["calibration"]["duration_seconds"]) - (timestamp - calibration_start)
                calibration_text = f"请保持正常坐姿，校准剩余：{max(0, int(remaining))} 秒"
                if timestamp - calibration_start >= float(config["calibration"]["duration_seconds"]):
                    calibration_result = calibration_collector.finalize()
                    calibrating = False
                    calibration_text = calibration_result.quality_message
            else:
                calibration_text = calibration_result.quality_message if not calibration_result.quality_ok else ""

            state = state_machine.update(features, metrics, timestamp, calibrating=calibrating)

            if recording and logger is not None:
                logger.log(
                    state=state,
                    features=features,
                    metrics=metrics,
                    camera_fps=camera_fps,
                    yolo_latency_ms=features.yolo_latency_ms,
                )

            display_frame = camera.make_display_frame(frame)
            display_frame = draw_overlay(
                frame=display_frame,
                state=state,
                face_observation=face_observation,
                features=features,
                metrics=metrics,
                camera_fps=camera_fps,
                yolo_fps=yolo_fps,
                calibration_text=calibration_text,
                debug=debug,
                mirror=bool(camera_cfg["mirror"]),
            )
            cv2.imshow("Drowsiness Signal Observer", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                calibration_collector.reset()
                calibration_result = _default_calibration_result(config)
                calibration_start = timestamp
                calibrating = calibration_enabled
            if key == ord("d"):
                debug = not debug
            if key == ord("r"):
                recording = not recording

            frame_id += 1

    except FaceLandmarkerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if yolo_worker is not None:
            yolo_worker.close()
        if landmarker is not None:
            landmarker.close()
        if logger is not None:
            logger.close()
        camera.release()
        camera.close_windows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
