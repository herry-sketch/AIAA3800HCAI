"""
OCR utilities and shared text-box data structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class TextBox:
    text: str
    bbox: list[float]
    confidence: float
    source: str

    @property
    def x_min(self) -> float:
        return self.bbox[0]

    @property
    def y_min(self) -> float:
        return self.bbox[1]

    @property
    def x_max(self) -> float:
        return self.bbox[2]

    @property
    def y_max(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def word_count(self) -> int:
        return len(self.text.replace("•", " ").split())


class OCREngine:
    """Lazy EasyOCR wrapper used when PDF embedded text is unavailable."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self.languages = languages or ["en", "ch_sim"]
        self.gpu = gpu
        self.reader: Any | None = None

    def _get_reader(self) -> Any:
        if self.reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError("EasyOCR is required for OCR fallback: pip install easyocr") from exc
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self.reader

    def extract_boxes(self, image_path: str, min_confidence: float = 0.25) -> list[TextBox]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        image = Image.open(path)
        image_width, image_height = image.size
        results = self._get_reader().readtext(
            str(path),
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5,
        )

        boxes: list[TextBox] = []
        for points, text, confidence in results:
            cleaned_text = " ".join(str(text).split())
            if not cleaned_text or float(confidence) < min_confidence:
                continue

            x_values = [float(point[0]) for point in points]
            y_values = [float(point[1]) for point in points]
            bbox = [
                clamp(min(x_values) / image_width),
                clamp(min(y_values) / image_height),
                clamp(max(x_values) / image_width),
                clamp(max(y_values) / image_height),
            ]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            boxes.append(TextBox(text=cleaned_text, bbox=bbox, confidence=float(confidence), source="ocr"))

        return sorted(boxes, key=lambda box: (box.y_min, box.x_min))

    def extract_text(self, image_path: str) -> str:
        return "\n".join(box.text for box in self.extract_boxes(image_path)).strip()

