"""
PDF loading, metadata management, slide rendering, and embedded text extraction.
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError("PyMuPDF is required: pip install PyMuPDF") from exc

from .ocr import TextBox, clamp


class SlideParser:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.uploaded_dir = self.data_dir / "uploaded_decks"
        self.images_dir = self.data_dir / "slide_images"
        self.metadata_file = self.data_dir / "deck_metadata.json"
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, dict[str, Any]] = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_file.exists():
            return {}
        try:
            with self.metadata_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid deck metadata JSON: {self.metadata_file}") from exc
        if not isinstance(data, dict):
            raise ValueError("Deck metadata must be a JSON object")
        return data

    def _save_metadata(self) -> None:
        tmp_path = self.metadata_file.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)
        tmp_path.replace(self.metadata_file)

    def load_deck(self, pdf_path: str) -> str:
        source_path = Path(pdf_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"PDF path is not a file: {source_path}")
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {source_path.suffix}")

        try:
            document = fitz.open(str(source_path))
            page_count = document.page_count
            document.close()
        except Exception as exc:
            raise ValueError(f"Unable to open PDF: {source_path}") from exc

        if page_count <= 0:
            raise ValueError("PDF has no pages")

        deck_id = uuid.uuid4().hex[:12]
        stored_pdf_path = self.uploaded_dir / f"{deck_id}.pdf"
        shutil.copy2(source_path, stored_pdf_path)

        self.metadata[deck_id] = {
            "deck_id": deck_id,
            "original_name": source_path.name,
            "pdf_path": str(stored_pdf_path),
            "page_count": page_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata()
        return deck_id

    def render_slide(self, deck_id: str, slide_id: int, dpi: int = 250) -> str:
        deck_info = self.metadata.get(deck_id)
        if deck_info is None:
            raise ValueError(f"Unknown deck_id: {deck_id}")
        if not isinstance(slide_id, int):
            raise TypeError("slide_id must be an integer")
        if dpi <= 0:
            raise ValueError("dpi must be positive")

        page_count = int(deck_info["page_count"])
        if slide_id < 1 or slide_id > page_count:
            raise ValueError(f"slide_id out of range: {slide_id}; page_count={page_count}")

        pdf_path = Path(str(deck_info["pdf_path"]))
        if not pdf_path.exists():
            raise FileNotFoundError(f"Stored PDF is missing: {pdf_path}")

        image_path = self.images_dir / f"{deck_id}_slide_{slide_id:03d}.png"
        document = fitz.open(str(pdf_path))
        try:
            page = document.load_page(slide_id - 1)
            zoom = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pixmap.save(str(image_path))
        finally:
            document.close()
        return str(image_path)

    def extract_pdf_text_boxes(self, deck_id: str, slide_id: int) -> list[TextBox]:
        deck_info = self.metadata.get(deck_id)
        if deck_info is None:
            raise ValueError(f"Unknown deck_id: {deck_id}")

        pdf_path = Path(str(deck_info["pdf_path"]))
        document = fitz.open(str(pdf_path))
        boxes: list[TextBox] = []
        try:
            page = document.load_page(slide_id - 1)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = []
                    for span in line.get("spans", []):
                        text = " ".join(str(span.get("text", "")).split())
                        bbox = span.get("bbox")
                        if text and bbox:
                            spans.append((text, bbox))
                    if not spans:
                        continue

                    text = " ".join(item[0] for item in spans)
                    x_min = min(float(item[1][0]) for item in spans)
                    y_min = min(float(item[1][1]) for item in spans)
                    x_max = max(float(item[1][2]) for item in spans)
                    y_max = max(float(item[1][3]) for item in spans)
                    bbox = [
                        clamp(x_min / page_width),
                        clamp(y_min / page_height),
                        clamp(x_max / page_width),
                        clamp(y_max / page_height),
                    ]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    boxes.append(TextBox(text=text, bbox=bbox, confidence=1.0, source="pdf_text"))
        finally:
            document.close()

        return sorted(boxes, key=lambda box: (box.y_min, box.x_min))

    def get_page_count(self, deck_id: str) -> int:
        deck_info = self.metadata.get(deck_id)
        if deck_info is None:
            raise ValueError(f"Unknown deck_id: {deck_id}")
        return int(deck_info["page_count"])

    def get_deck_info(self, deck_id: str) -> dict[str, Any] | None:
        return self.metadata.get(deck_id)


def load_deck(pdf_path: str) -> str:
    return SlideParser().load_deck(pdf_path)


def render_slide(deck_id: str, slide_id: int) -> str:
    return SlideParser().render_slide(deck_id, slide_id)

