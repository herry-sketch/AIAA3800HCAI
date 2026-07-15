from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_config_path, resolve_runtime_paths
from src.face_landmarker import FaceLandmarkerError, MediaPipeFaceLandmarker
from src.yolo_classifier import YoloClassifier, YoloClassifierError, resolve_device_config


def _check_task_model(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(32)
    if prefix.startswith(b"PLACEHOLDER"):
        raise RuntimeError(
            "face_landmarker.task is still the repository placeholder. Replace it with the real MediaPipe task model."
        )


def _print_cuda_info(device: int | str, half: bool) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        print("torch import: unavailable")
        print(f"configured device: {device}")
        print(f"configured half: {half}")
        return

    cuda_available = torch.cuda.is_available()
    print(f"cuda_available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"gpu_name: {gpu_name}")
    else:
        print("gpu_name: N/A")
    print(f"configured device: {device}")
    print(f"configured half: {half}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize MediaPipe and YOLO without using the camera")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, PROJECT_ROOT)
    config = resolve_runtime_paths(load_config(config_path), config_path.parent)
    task_path = Path(str(config["models"]["face_landmarker_path"]))

    landmarker = None
    try:
        _check_task_model(task_path)
        print(f"face_landmarker.task: OK ({task_path})")

        landmarker = MediaPipeFaceLandmarker(
            model_path=str(task_path),
            mediapipe_config=config["mediapipe"],
            face_crop_config=config["face_crop"],
        )
        print("MediaPipe Face Landmarker: initialized")

        device, half = resolve_device_config(
            requested_device=config["models"].get("yolo_device"),
            requested_half=config["models"].get("yolo_half"),
        )
        _print_cuda_info(device, half)

        classifier = YoloClassifier(
            local_path=str(config["models"]["yolo_local_path"]),
            repo_id=str(config["models"]["yolo_repo_id"]),
            filename=str(config["models"]["yolo_filename"]),
            imgsz=int(config["models"]["yolo_imgsz"]),
            device=device,
            half=half,
        )
        print(f"YOLO class map: {classifier.class_names}")
        print("Model initialization check: PASS")
        return 0
    except (FaceLandmarkerError, YoloClassifierError, FileNotFoundError, RuntimeError) as exc:
        print(f"Model initialization check: FAIL - {exc}", file=sys.stderr)
        return 1
    finally:
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    raise SystemExit(main())
