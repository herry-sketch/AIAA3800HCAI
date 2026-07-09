"""
Small demo for the modular Slide & AOI system.
"""
from __future__ import annotations

import argparse
import json

from modules.slide.aoi_manager import AOIManager
from modules.slide.slide_parser import SlideParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--slide-id", type=int, default=1)
    args = parser.parse_args()

    slide_parser = SlideParser()
    aoi_manager = AOIManager()

    deck_id = slide_parser.load_deck(args.pdf_path)
    slide_data = aoi_manager.process_slide(deck_id, args.slide_id)

    print(json.dumps({"deck_id": deck_id, "slide": slide_data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
