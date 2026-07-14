"""
AOI generation, persistence, retrieval, and manual correction.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .llm_aoi import LLMAOIGenerator
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
    llm_aoi_status: str = "not_requested"
    llm_aoi_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "slide_id": self.slide_id,
            "slide_image_path": self.slide_image_path,
            "ocr_text": self.ocr_text,
            "slide_text": self.ocr_text,
            "text_source": self.text_source,
            "auto_aoi_method": self.auto_aoi_method,
            "llm_aoi_status": self.llm_aoi_status,
            "aois": [aoi.to_dict() for aoi in self.aois],
        }
        if self.llm_aoi_error:
            data["llm_aoi_error"] = self.llm_aoi_error
        return data


class AOIManager:
    def __init__(self, data_dir: str = "data", use_llm_aoi: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.use_llm_aoi = use_llm_aoi
        self.llm_aoi_generator = LLMAOIGenerator() if use_llm_aoi else None
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

    def process_slide(
        self,
        deck_id: str,
        slide_id: int,
        dpi: int = 250,
        use_llm_aoi: bool | None = None,
    ) -> dict[str, Any]:
        parser = SlideParser(str(self.data_dir))
        image_path = parser.render_slide(deck_id, slide_id, dpi=dpi)
        pdf_text_boxes = parser.extract_pdf_text_boxes(deck_id, slide_id)
        pdf_text = "\n".join(box.text for box in pdf_text_boxes).strip()

        if len(pdf_text) >= 30:
            text_boxes = pdf_text_boxes
            text_source = "pdf_text"
            auto_aois = self.build_pdf_semantic_aois(text_boxes)
            auto_aoi_method = "pdf_text_semantic"
            image_boxes = parser.extract_pdf_image_boxes(deck_id, slide_id)
            image_text_boxes = self.extract_image_text_boxes(image_boxes, image_path)
            image_text_boxes = self.merge_text_boxes([], image_text_boxes)
            image_text_boxes = [
                box
                for box in image_text_boxes
                if not self._text_box_is_duplicate(box, pdf_text_boxes)
            ]
            if image_text_boxes:
                auto_aois.extend(self.build_image_region_aois(image_boxes, image_text_boxes))
                auto_aoi_method += "_with_image_ocr"
        else:
            text_boxes = OCREngine().extract_boxes(image_path)
            text_source = "ocr"
            auto_aois = self.build_text_block_aois(text_boxes)
            auto_aoi_method = "ocr_text_block"

        slide_text = "\n".join(box.text for box in text_boxes).strip()
        rule_aois = self.generate_rule_aois(slide_text)
        self.populate_rule_aoi_text(rule_aois, auto_aois)
        output_aois = auto_aois
        llm_status = "not_requested"
        llm_error = None

        should_use_llm = self.use_llm_aoi if use_llm_aoi is None else use_llm_aoi
        if should_use_llm:
            try:
                llm_aois = self.build_llm_guided_aois(image_path, slide_text, rule_aois, auto_aois)
                output_aois = self.reconcile_llm_aois(llm_aois, auto_aois)
                auto_aoi_method = f"llm_guided_with_{auto_aoi_method}_fallback"
                llm_status = "used"
            except Exception as exc:
                llm_status = "fallback_used"
                llm_error = str(exc)

        slide_data = SlideAOIData(
            slide_id=slide_id,
            slide_image_path=image_path,
            ocr_text=slide_text,
            aois=output_aois or rule_aois,
            text_source=text_source,
            auto_aoi_method=auto_aoi_method,
            llm_aoi_status=llm_status,
            llm_aoi_error=llm_error,
        )
        return self.save_slide_data(deck_id, slide_data)

    def extract_image_text_boxes(
        self,
        image_boxes: list[list[float]],
        image_path: str,
    ) -> list[TextBox]:
        """OCR embedded image regions without making OCR a hard dependency."""
        try:
            if not image_boxes:
                return []
            ocr = OCREngine()
            return [
                text_box
                for image_box in image_boxes
                for text_box in ocr.extract_region_boxes(image_path, image_box)
            ]
        except Exception:
            return []

    def build_image_region_aois(
        self,
        image_boxes: list[list[float]],
        text_boxes: list[TextBox],
    ) -> list[AOI]:
        """Represent each embedded image as one visual AOI, never as OCR fragments."""
        visual_aois: list[AOI] = []
        for image_bbox in image_boxes:
            region_text_boxes = [
                box
                for box in text_boxes
                if self._bbox_center_inside(box.bbox, image_bbox)
            ]
            if not region_text_boxes:
                continue
            region_text_boxes.sort(key=lambda box: (box.y_min, box.x_min))
            text = "\n".join(box.text for box in region_text_boxes)
            lowered = text.casefold()
            code_markers = re.findall(
                r"(?:#|=|\b(?:def|from|import|print|return)\b)",
                lowered,
            )
            aspect_ratio = (image_bbox[2] - image_bbox[0]) / max(
                image_bbox[3] - image_bbox[1],
                1e-6,
            )
            if len(code_markers) >= 2:
                aoi_type = "code"
            elif aspect_ratio >= 2.5:
                aoi_type = "table"
            else:
                aoi_type = "figure"
            visual_aois.append(
                AOI(
                    aoi_id=f"image_region_{len(visual_aois) + 1}",
                    bbox=list(image_bbox),
                    type=aoi_type,
                    text=text,
                    source="ocr_image",
                    group_confidence=min(
                        box.confidence for box in region_text_boxes
                    ),
                )
            )
        return visual_aois

    @classmethod
    def merge_text_boxes(
        cls,
        primary: list[TextBox],
        supplementary: list[TextBox],
    ) -> list[TextBox]:
        """Merge positioned text while preferring the primary PDF extraction."""
        merged = list(primary)
        for candidate in supplementary:
            if not cls._text_box_is_duplicate(candidate, merged):
                merged.append(candidate)
        return sorted(merged, key=lambda box: (box.y_min, box.x_min))

    @classmethod
    def _text_box_is_duplicate(cls, candidate: TextBox, existing: list[TextBox]) -> bool:
        candidate_text = cls._normalize_text(candidate.text)
        for box in existing:
            existing_text = cls._normalize_text(box.text)
            if candidate_text and candidate_text == existing_text:
                return True
            if cls._bbox_iou(candidate.bbox, box.bbox) >= 0.70:
                return True
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())

    @staticmethod
    def _bbox_iou(first: list[float], second: list[float]) -> float:
        width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        intersection = width * height
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

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

    def merge_pdf_wrapped_aois(self, aois: list[AOI]) -> list[AOI]:
        """Join visual line wraps without joining separate rows or columns."""
        merged: list[AOI] = []
        for candidate in sorted(aois, key=lambda aoi: (aoi.bbox[1], aoi.bbox[0])):
            current = next(
                (
                    previous
                    for previous in reversed(merged)
                    if self._pdf_lines_are_continuous(previous, candidate)
                ),
                None,
            )
            if current is not None:
                current.text = f"{current.text.rstrip()} {candidate.text.lstrip()}"
                current.bbox = [
                    min(current.bbox[0], candidate.bbox[0]),
                    min(current.bbox[1], candidate.bbox[1]),
                    max(current.bbox[2], candidate.bbox[2]),
                    max(current.bbox[3], candidate.bbox[3]),
                ]
                current.group_confidence = min(
                    current.group_confidence or 0.62,
                    candidate.group_confidence or 0.62,
                )
            else:
                merged.append(candidate)
        for index, aoi in enumerate(merged, start=1):
            aoi.aoi_id = f"pdf_semantic_block_{index}"
        return merged

    @staticmethod
    def _pdf_lines_are_continuous(first: AOI, second: AOI) -> bool:
        if first.type != "text" or second.type != "text":
            return False
        first_text = first.text.strip()
        second_text = second.text.strip()
        if not first_text or not second_text:
            return False
        if second_text.startswith(("•", "o ", "*")):
            return False
        if second_text.casefold().startswith("method "):
            return False
        if first_text.endswith((".", "!", "?", "]")):
            return False
        same_column = abs(first.bbox[0] - second.bbox[0]) <= 0.025
        vertical_gap = max(0.0, second.bbox[1] - first.bbox[3])
        return same_column and vertical_gap <= 0.025

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
        return self.merge_pdf_wrapped_aois(aois)

    def build_llm_guided_aois(
        self,
        image_path: str,
        slide_text: str,
        rule_aois: list[AOI],
        text_aois: list[AOI],
    ) -> list[AOI]:
        generator = self.llm_aoi_generator or LLMAOIGenerator()
        llm_aoi_dicts = generator.generate(
            image_path=image_path,
            slide_text=slide_text,
            rule_aois=[aoi.to_dict() for aoi in rule_aois],
            text_aois=[aoi.to_dict() for aoi in text_aois],
        )

        llm_aois: list[AOI] = []
        for item in llm_aoi_dicts:
            bbox = [float(value) for value in item["bbox"]]
            self._validate_bbox(bbox)
            llm_aois.append(
                AOI(
                    aoi_id=str(item["aoi_id"]),
                    bbox=bbox,
                    type=str(item["type"]),
                    text=str(item.get("text", "")),
                    source="llm_guided",
                    group_confidence=float(item.get("group_confidence", 0.70)),
                    include_in_learning=bool(item.get("include_in_learning", True)),
                )
            )
        return llm_aois

    def reconcile_llm_aois(
        self,
        llm_aois: list[AOI],
        grounding_aois: list[AOI],
        minimum_text_coverage: float = 0.45,
    ) -> list[AOI]:
        """Validate one flat LLM result against deterministic PDF/OCR anchors."""
        groundings_by_text: dict[str, list[AOI]] = {}
        for grounding in grounding_aois:
            normalized = self._normalize_text(grounding.text)
            if normalized:
                groundings_by_text.setdefault(normalized, []).append(grounding)

        text_types = {"title", "text", "caption", "footer", "axis_label"}
        for aoi in llm_aois:
            matches = groundings_by_text.get(self._normalize_text(aoi.text), [])
            if aoi.type in text_types and len(matches) == 1:
                aoi.bbox = list(matches[0].bbox)

        candidates = [aoi for aoi in llm_aois if self._bbox_area(aoi.bbox) < 0.90]
        candidates.sort(key=lambda aoi: aoi.group_confidence or 0.0, reverse=True)
        resolved: list[AOI] = []
        for candidate in candidates:
            has_conflict = any(
                self._same_aoi_category(candidate, existing)
                and self._bbox_iou(candidate.bbox, existing.bbox) >= 0.85
                for existing in resolved
            )
            if not has_conflict:
                resolved.append(candidate)
        resolved.sort(key=lambda aoi: (aoi.bbox[1], aoi.bbox[0]))

        grounding_tokens = self._learning_tokens(grounding_aois)
        if len(grounding_tokens) >= 8:
            llm_tokens = self._learning_tokens(resolved)
            coverage = len(grounding_tokens & llm_tokens) / len(grounding_tokens)
            if coverage < minimum_text_coverage:
                raise ValueError(
                    f"LLM AOI text coverage too low: {coverage:.1%}; "
                    f"required {minimum_text_coverage:.0%}"
                )
        if not resolved:
            raise ValueError("LLM AOI reconciliation produced no usable AOIs")
        for index, aoi in enumerate(resolved, start=1):
            aoi.aoi_id = f"llm_aoi_{index}"
        return resolved

    @classmethod
    def _learning_tokens(cls, aois: list[AOI]) -> set[str]:
        tokens: set[str] = set()
        for aoi in aois:
            if aoi.type == "footer":
                continue
            tokens.update(cls._normalize_text(aoi.text).split())
        return tokens

    @staticmethod
    def _same_aoi_category(first: AOI, second: AOI) -> bool:
        visual_types = {"code", "diagram", "figure", "table", "formula"}
        return (first.type in visual_types) == (second.type in visual_types)

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
        left_text = left.text.strip()
        if (
            dy > 0.035
            or gap < 0.0
            or gap > 0.30
            or height_ratio < 0.55
            or left.word_count > 3
            or left_text.endswith(":")
        ):
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
            include_in_learning=aoi_type != "footer",
        )

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
