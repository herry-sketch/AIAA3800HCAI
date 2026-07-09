"""
AOI generation, persistence, retrieval, and manual correction.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .ocr import OCREngine, TextBox, clamp
from .slide_parser import SlideParser


@dataclass
class AOI:
    aoi_id: str
    bbox: list[float]
    type: str
    text: str = ""
    source: str = "rule"
    group_confidence: float | None = None
    children: list[dict[str, Any]] | None = None
    include_in_learning: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class SlideAOIData:
    slide_id: int
    slide_image_path: str
    ocr_text: str
    aois: list[AOI]
    text_source: str
    auto_aoi_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "slide_image_path": self.slide_image_path,
            "ocr_text": self.ocr_text,
            "slide_text": self.ocr_text,
            "text_source": self.text_source,
            "auto_aoi_method": self.auto_aoi_method,
            "aois": [aoi.to_dict() for aoi in self.aois],
        }


class AOIManager:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.manifest_file = self.data_dir / "aoi_manifest.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_file.exists():
            return {}
        try:
            with self.manifest_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid AOI manifest JSON: {self.manifest_file}") from exc
        if not isinstance(data, dict):
            raise ValueError("AOI manifest must be a JSON object")
        return data

    def _save_manifest(self) -> None:
        tmp_path = self.manifest_file.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self.manifest, file, ensure_ascii=False, indent=2)
        tmp_path.replace(self.manifest_file)

    @staticmethod
    def _slide_key(deck_id: str, slide_id: int) -> str:
        return f"{deck_id}:{slide_id}"

    def process_slide(self, deck_id: str, slide_id: int, dpi: int = 250) -> dict[str, Any]:
        parser = SlideParser(str(self.data_dir))
        image_path = parser.render_slide(deck_id, slide_id, dpi=dpi)
        pdf_text_boxes = parser.extract_pdf_text_boxes(deck_id, slide_id)
        pdf_text = "\n".join(box.text for box in pdf_text_boxes).strip()

        if len(pdf_text) >= 30:
            text_boxes = pdf_text_boxes
            text_source = "pdf_text"
            auto_aois = self.build_pdf_semantic_aois(text_boxes)
            auto_aoi_method = "pdf_text_semantic"
        else:
            text_boxes = OCREngine().extract_boxes(image_path)
            text_source = "ocr"
            auto_aois = self.build_text_block_aois(text_boxes)
            auto_aoi_method = "ocr_text_block"

        slide_text = "\n".join(box.text for box in text_boxes).strip()
        rule_aois = self.generate_rule_aois(slide_text)
        self.populate_rule_aoi_text(rule_aois, auto_aois)
        slide_data = SlideAOIData(
            slide_id=slide_id,
            slide_image_path=image_path,
            ocr_text=slide_text,
            aois=rule_aois + auto_aois,
            text_source=text_source,
            auto_aoi_method=auto_aoi_method,
        )
        return self.save_slide_data(deck_id, slide_data)

    def generate_rule_aois(self, slide_text: str = "") -> list[AOI]:
        return [
            AOI("title", [0.05, 0.05, 0.95, 0.18], "title", source="rule"),
            AOI("top_region", [0.05, 0.18, 0.95, 0.35], "text", source="rule"),
            AOI("left_block", [0.05, 0.35, 0.48, 0.85], "text", source="rule"),
            AOI("right_block", [0.52, 0.35, 0.95, 0.85], "mixed", source="rule"),
            AOI("right_visual_region", [0.50, 0.43, 0.96, 0.80], "figure", source="rule"),
            AOI("bottom_region", [0.05, 0.85, 0.95, 0.98], "caption", source="rule"),
            AOI("whole_slide", [0.0, 0.0, 1.0, 1.0], "whole_slide", text=slide_text, source="rule"),
        ]

    def populate_rule_aoi_text(self, rule_aois: list[AOI], auto_aois: list[AOI]) -> None:
        for rule_aoi in rule_aois:
            if rule_aoi.aoi_id == "whole_slide":
                continue
            if rule_aoi.type == "figure":
                continue
            texts = []
            for auto_aoi in auto_aois:
                if auto_aoi.type == "footer":
                    continue
                if self._bbox_center_inside(auto_aoi.bbox, rule_aoi.bbox):
                    texts.append(auto_aoi.text)
            if texts:
                rule_aoi.text = "\n".join(texts)
                if rule_aoi.aoi_id == "right_block" and any(text.strip() for text in texts):
                    rule_aoi.type = "mixed"

    def build_text_block_aois(self, text_boxes: list[TextBox]) -> list[AOI]:
        blocks: list[list[TextBox]] = []
        for box in sorted(text_boxes, key=lambda item: (item.y_min, item.x_min)):
            target_block = self._find_best_block(box, blocks)
            if target_block is None:
                blocks.append([box])
            else:
                target_block.append(box)

        aois: list[AOI] = []
        for block in blocks:
            block = sorted(block, key=lambda item: (item.y_min, item.x_min))
            bbox = self._merged_bbox(block, padding=0.01)
            if self._bbox_area(bbox) < 0.002:
                continue
            aois.append(
                AOI(
                    aoi_id=f"ocr_text_block_{len(aois) + 1}",
                    bbox=bbox,
                    type="title" if bbox[1] < 0.20 and len(block) <= 3 else "text",
                    text="\n".join(item.text for item in block),
                    source="ocr",
                )
            )
        return aois

    def build_pdf_semantic_aois(self, text_boxes: list[TextBox], threshold: float = 0.72) -> list[AOI]:
        pdf_boxes = [box for box in text_boxes if box.source == "pdf_text"]
        used: set[int] = set()
        aois: list[AOI] = []
        sorted_items = sorted(enumerate(pdf_boxes), key=lambda item: (item[1].y_min, item[1].x_min))

        for left_index, left_box in sorted_items:
            if left_index in used:
                continue

            special_type = self._special_pdf_box_type(left_box)
            if special_type is not None:
                used.add(left_index)
                aois.append(
                    self._make_semantic_aoi(
                        f"pdf_semantic_block_{len(aois) + 1}",
                        [left_box],
                        special_type,
                        0.95,
                    )
                )
                continue

            best_index = None
            best_confidence = 0.0
            for right_index, right_box in sorted_items:
                if right_index == left_index or right_index in used:
                    continue
                if self._special_pdf_box_type(right_box) is not None or right_box.x_min <= left_box.x_max:
                    continue
                confidence = self._semantic_pair_confidence(left_box, right_box)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_index = right_index

            if best_index is not None and best_confidence >= threshold:
                used.add(left_index)
                used.add(best_index)
                aois.append(
                    self._make_semantic_aoi(
                        f"pdf_semantic_block_{len(aois) + 1}",
                        [left_box, pdf_boxes[best_index]],
                        "text",
                        best_confidence,
                    )
                )
            else:
                used.add(left_index)
                aoi_type = "title" if left_box.y_min < 0.20 else "text"
                aois.append(
                    self._make_semantic_aoi(
                        f"pdf_semantic_block_{len(aois) + 1}",
                        [left_box],
                        aoi_type,
                        0.62,
                    )
                )
        return aois

    def save_slide_data(self, deck_id: str, slide_data: SlideAOIData) -> dict[str, Any]:
        for aoi in slide_data.aois:
            self._validate_bbox(aoi.bbox)
        data = slide_data.to_dict()
        self.manifest[self._slide_key(deck_id, slide_data.slide_id)] = data
        self._save_manifest()
        return data

    def get_slide_aois(self, deck_id: str, slide_id: int) -> list[dict[str, Any]]:
        slide_data = self._ensure_slide_data(deck_id, slide_id)
        return list(slide_data["aois"])

    def get_slide_text(self, deck_id: str, slide_id: int) -> str:
        return str(self._ensure_slide_data(deck_id, slide_id).get("ocr_text", ""))

    def update_aoi(
        self,
        deck_id: str,
        slide_id: int,
        aoi_id: str,
        bbox: list[float],
        aoi_type: str,
        text: str,
    ) -> dict[str, Any]:
        self._validate_bbox(bbox)
        slide_data = self._ensure_slide_data(deck_id, slide_id)
        for aoi in slide_data["aois"]:
            if aoi["aoi_id"] == aoi_id:
                aoi["bbox"] = [float(value) for value in bbox]
                aoi["type"] = aoi_type
                aoi["text"] = text
                aoi["source"] = "manual"
                self.manifest[self._slide_key(deck_id, slide_id)] = slide_data
                self._save_manifest()
                return aoi
        raise ValueError(f"AOI does not exist: {aoi_id}")

    def add_aoi(
        self,
        deck_id: str,
        slide_id: int,
        aoi_id: str,
        bbox: list[float],
        aoi_type: str,
        text: str = "",
    ) -> dict[str, Any]:
        self._validate_bbox(bbox)
        slide_data = self._ensure_slide_data(deck_id, slide_id)
        if any(aoi["aoi_id"] == aoi_id for aoi in slide_data["aois"]):
            raise ValueError(f"AOI already exists: {aoi_id}")
        new_aoi = AOI(aoi_id, [float(value) for value in bbox], aoi_type, text, source="manual").to_dict()
        slide_data["aois"].append(new_aoi)
        self.manifest[self._slide_key(deck_id, slide_id)] = slide_data
        self._save_manifest()
        return new_aoi

    def delete_aoi(self, deck_id: str, slide_id: int, aoi_id: str) -> None:
        slide_data = self._ensure_slide_data(deck_id, slide_id)
        original_count = len(slide_data["aois"])
        slide_data["aois"] = [aoi for aoi in slide_data["aois"] if aoi["aoi_id"] != aoi_id]
        if len(slide_data["aois"]) == original_count:
            raise ValueError(f"AOI does not exist: {aoi_id}")
        self.manifest[self._slide_key(deck_id, slide_id)] = slide_data
        self._save_manifest()

    def get_gaze_payload(self, deck_id: str, slide_id: int) -> dict[str, Any]:
        aois = [
            {"aoi_id": aoi["aoi_id"], "bbox": aoi["bbox"], "type": aoi["type"]}
            for aoi in self.get_slide_aois(deck_id, slide_id)
            if aoi.get("type") != "footer"
        ]
        return {"slide_id": slide_id, "aois": aois}

    def get_tutor_payload(self, deck_id: str, slide_id: int) -> dict[str, Any]:
        slide_data = self._ensure_slide_data(deck_id, slide_id)
        aois = [
            {"aoi_id": aoi["aoi_id"], "type": aoi["type"], "text": aoi.get("text", "")}
            for aoi in slide_data["aois"]
            if aoi.get("text") and aoi.get("type") != "footer"
        ]
        return {"slide_id": slide_id, "ocr_text": slide_data.get("ocr_text", ""), "aois": aois}

    def _ensure_slide_data(self, deck_id: str, slide_id: int) -> dict[str, Any]:
        key = self._slide_key(deck_id, slide_id)
        if key not in self.manifest:
            self.process_slide(deck_id, slide_id)
        return self.manifest[key]

    def _semantic_pair_confidence(self, left: TextBox, right: TextBox) -> float:
        dy = abs(left.y_center - right.y_center)
        gap = right.x_min - left.x_max
        height_ratio = min(left.height, right.height) / max(left.height, right.height, 1e-6)
        if dy > 0.035 or gap < 0.0 or gap > 0.30 or height_ratio < 0.55:
            return 0.0

        row_score = 1.0 - min(dy / 0.035, 1.0)
        gap_score = 1.0 - min(gap / 0.30, 1.0)
        height_score = height_ratio
        left_short = left.word_count <= 3
        right_not_shorter = right.word_count >= left.word_count
        right_phrase_like = right.word_count >= 2 or len(right.text) >= 10
        both_very_short = left.word_count <= 2 and right.word_count <= 2

        if left_short and right_not_shorter and right_phrase_like:
            role_score = 1.0
        elif both_very_short:
            role_score = 0.35
        elif left_short:
            role_score = 0.70
        else:
            role_score = 0.45

        return round(0.35 * row_score + 0.25 * gap_score + 0.20 * height_score + 0.20 * role_score, 3)

    def _make_semantic_aoi(self, aoi_id: str, boxes: list[TextBox], aoi_type: str, confidence: float) -> AOI:
        bbox = self._merged_bbox(boxes, padding=0.006)
        return AOI(
            aoi_id=aoi_id,
            bbox=bbox,
            type=aoi_type,
            text="\n".join(box.text for box in boxes),
            source="pdf_text_semantic",
            group_confidence=round(float(confidence), 3),
            children=[self._text_box_child(box) for box in boxes],
            include_in_learning=aoi_type != "footer",
        )

    @staticmethod
    def _text_box_child(box: TextBox) -> dict[str, Any]:
        return {"text": box.text, "bbox": box.bbox, "source": box.source, "confidence": box.confidence}

    @staticmethod
    def _special_pdf_box_type(box: TextBox) -> str | None:
        if box.y_min > 0.82:
            return "footer"
        if box.y_min < 0.20:
            return "title"
        if box.width < 0.055 and box.height > 0.12:
            return "axis_label"
        return None

    def _find_best_block(self, box: TextBox, blocks: list[list[TextBox]]) -> list[TextBox] | None:
        best_block = None
        best_score = float("inf")
        for block in blocks:
            block_bbox = self._merged_bbox(block, padding=0.0)
            vertical_gap = box.y_min - block_bbox[3]
            if vertical_gap < -0.012:
                continue
            same_column = self._horizontal_overlap(box.bbox, block_bbox) >= 0.25
            left_aligned = abs(box.x_min - block_bbox[0]) <= 0.07
            close_y = 0.0 <= vertical_gap <= max(0.04, 1.8 * self._median_height(block))
            if close_y and (same_column or left_aligned):
                score = vertical_gap + abs(box.x_min - block_bbox[0]) * 0.4
                if score < best_score:
                    best_score = score
                    best_block = block
        return best_block

    def _merged_bbox(self, boxes: list[TextBox], padding: float = 0.0) -> list[float]:
        return [
            clamp(min(box.x_min for box in boxes) - padding),
            clamp(min(box.y_min for box in boxes) - padding),
            clamp(max(box.x_max for box in boxes) + padding),
            clamp(max(box.y_max for box in boxes) + padding),
        ]

    @staticmethod
    def _bbox_center_inside(inner: list[float], outer: list[float]) -> bool:
        x_center = (inner[0] + inner[2]) / 2
        y_center = (inner[1] + inner[3]) / 2
        return outer[0] <= x_center <= outer[2] and outer[1] <= y_center <= outer[3]

    @staticmethod
    def _horizontal_overlap(a: list[float], b: list[float]) -> float:
        overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        min_width = max(1e-6, min(a[2] - a[0], b[2] - b[0]))
        return overlap / min_width

    @staticmethod
    def _median_height(block: list[TextBox]) -> float:
        heights = sorted(box.height for box in block)
        return heights[len(heights) // 2]

    @staticmethod
    def _bbox_area(bbox: list[float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _validate_bbox(bbox: list[float]) -> None:
        if len(bbox) != 4:
            raise ValueError("bbox must have four values: [x_min, y_min, x_max, y_max]")
        x_min, y_min, x_max, y_max = [float(value) for value in bbox]
        if not all(0.0 <= value <= 1.0 for value in [x_min, y_min, x_max, y_max]):
            raise ValueError("bbox values must be between 0.0 and 1.0")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("bbox must satisfy x_min < x_max and y_min < y_max")


def get_slide_aois(deck_id: str, slide_id: int) -> list[dict[str, Any]]:
    return AOIManager().get_slide_aois(deck_id, slide_id)


def get_slide_text(deck_id: str, slide_id: int) -> str:
    return AOIManager().get_slide_text(deck_id, slide_id)


def update_aoi(deck_id: str, slide_id: int, aoi_id: str, bbox: list[float], aoi_type: str, text: str) -> dict[str, Any]:
    return AOIManager().update_aoi(deck_id, slide_id, aoi_id, bbox, aoi_type, text)


def add_aoi(deck_id: str, slide_id: int, aoi_id: str, bbox: list[float], aoi_type: str, text: str = "") -> dict[str, Any]:
    return AOIManager().add_aoi(deck_id, slide_id, aoi_id, bbox, aoi_type, text)


def delete_aoi(deck_id: str, slide_id: int, aoi_id: str) -> None:
    AOIManager().delete_aoi(deck_id, slide_id, aoi_id)
