from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraError, CameraStream
from src.config import load_config, resolve_config_path, resolve_runtime_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Read 30 frames from the configured camera and exit")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, PROJECT_ROOT)
    config = resolve_runtime_paths(load_config(config_path), config_path.parent)
    camera_cfg = config["camera"]
    camera = CameraStream(
        index=int(camera_cfg["index"]),
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        mirror=bool(camera_cfg["mirror"]),
    )

    success_frames = 0
    frame_width = None
    frame_height = None

    try:
        camera.open()
        for _ in range(30):
            ok, frame, _timestamp = camera.read()
            if ok and frame is not None:
                success_frames += 1
                frame_height, frame_width = frame.shape[:2]
        print(f"success_frames: {success_frames}/30")
        if frame_width is not None and frame_height is not None:
            print(f"actual_resolution: {frame_width}x{frame_height}")
        else:
            print("actual_resolution: N/A")
        return 0 if success_frames == 30 else 1
    except CameraError as exc:
        print(f"camera_smoke_test: FAIL - {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        camera.close_windows()


if __name__ == "__main__":
    raise SystemExit(main())
