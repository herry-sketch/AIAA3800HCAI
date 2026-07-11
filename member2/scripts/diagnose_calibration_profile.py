from __future__ import annotations

import argparse
from pathlib import Path

from diagnose_human_sensing_modules import inspect_calibration_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a 9-point calibration profile.")
    parser.add_argument("--calibration-profile", type=Path, required=True)
    args = parser.parse_args()
    return inspect_calibration_profile(args.calibration_profile)


if __name__ == "__main__":
    raise SystemExit(main())
