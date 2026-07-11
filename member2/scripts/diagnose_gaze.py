from __future__ import annotations

import argparse

from diagnose_human_sensing_modules import add_common_camera_args, run_camera_diagnostics, set_mediapipe_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose gaze-grid prediction.")
    add_common_camera_args(parser)
    args = parser.parse_args()
    args.module = "gaze"
    set_mediapipe_model(args.mediapipe_model)
    return run_camera_diagnostics(args)


if __name__ == "__main__":
    raise SystemExit(main())
