"""
Optional LLM/VLM-guided AOI generation.

This module is intentionally optional. If no OpenAI-compatible vision endpoint
is configured, the main AOI pipeline continues to use the existing rule-based
and PDF/OCR text-block fallback logic.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_AOI_TYPES = {
    "title",
    "text",
    "figure",
    "table",
    "formula",
    "caption",
    "footer",
    "axis_label",
    "mixed",
    "whole_slide",
}


@dataclass
class LLMAOIConfig:
    endpoint: str | None = None
    api_key: str | None = None
    model: str = "qwen2.5-vl-7b-instruct"
    timeout_sec: int = 90
    max_image_side: int = 1280

    @classmethod
    def from_env(cls) -> "LLMAOIConfig":
        return cls(
            endpoint=os.getenv("SLIDE_AOI_LLM_ENDPOINT"),
            api_key=os.getenv("SLIDE_AOI_LLM_API_KEY"),
            model=os.getenv("SLIDE_AOI_LLM_MODEL", "qwen2.5-vl-7b-instruct"),
            timeout_sec=int(os.getenv("SLIDE_AOI_LLM_TIMEOUT_SEC", "90")),
            max_image_side=int(os.getenv("SLIDE_AOI_LLM_MAX_IMAGE_SIDE", "1280")),
        )


class LLMAOIGenerator:
    """Calls an OpenAI-compatible vision chat endpoint and validates AOI JSON."""

    def __init__(self, config: LLMAOIConfig | None = None) -> None:
        self.config = config or LLMAOIConfig.from_env()

    def is_configured(self) -> bool:
        return bool(self.config.endpoint)

    def generate(
        self,
        image_path: str,
        slide_text: str,
        rule_aois: list[dict[str, Any]],
        text_aois: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.is_configured():
            raise RuntimeError("SLIDE_AOI_LLM_ENDPOINT is not configured")

        payload = self._build_payload(image_path, slide_text, rule_aois, text_aois)
        request = urllib.request.Request(
            str(self.config.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM AOI request failed: {exc}") from exc

        content = self._extract_message_content(raw)
        data = self._extract_json_object(content)
        aois = data.get("aois")
        if not isinstance(aois, list):
            raise ValueError("LLM AOI response must contain an 'aois' list")
        return self._validate_aois(aois)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(
        self,
        image_path: str,
        slide_text: str,
        rule_aois: list[dict[str, Any]],
        text_aois: list[dict[str, Any]],
    ) -> dict[str, Any]:
        image_url = self._image_data_url(image_path)
        prompt = self._prompt(slide_text, rule_aois, text_aois)
        return {
            "model": self.config.model,
            "temperature": 0.1,
            "max_tokens": 1600,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a slide layout analyst. Return only valid JSON. "
                        "All bounding boxes must use normalized coordinates [x_min, y_min, x_max, y_max]."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }

    def _prompt(self, slide_text: str, rule_aois: list[dict[str, Any]], text_aois: list[dict[str, Any]]) -> str:
        return (
            "Generate semantic AOIs for this lecture slide.\n\n"
            "Use the rule AOIs as guidance, but produce more meaningful learning units when possible. "
            "Prefer grouping title + subtitle, paired concept + explanation, figure + caption, table, formula, "
            "and diagram regions. Do not invent text that is not visible on the slide.\n\n"
            "Allowed AOI types: title, text, figure, table, formula, caption, footer, axis_label, mixed.\n"
            "Each AOI must include: aoi_id, bbox, type, text, confidence.\n"
            "Use aoi_id prefix 'llm_aoi_'. confidence must be between 0 and 1.\n"
            "Return exactly this JSON shape:\n"
            "{\"aois\":[{\"aoi_id\":\"llm_aoi_1\",\"bbox\":[0.1,0.1,0.5,0.3],"
            "\"type\":\"text\",\"text\":\"...\",\"confidence\":0.85}]}\n\n"
            f"Slide text:\n{slide_text[:5000]}\n\n"
            f"Rule AOIs:\n{json.dumps(rule_aois, ensure_ascii=False)}\n\n"
            f"Existing text AOIs from PDF/OCR:\n{json.dumps(text_aois, ensure_ascii=False)}\n"
        )

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Slide image does not exist: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _extract_message_content(raw_response: str) -> str:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                return "\n".join(str(part) for part in parts if part)
        return raw_response

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM AOI response did not contain JSON")
            return json.loads(match.group(0))

    def _validate_aois(self, aois: list[Any]) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []
        for index, item in enumerate(aois, start=1):
            if not isinstance(item, dict):
                continue

            bbox = item.get("bbox")
            if not self._valid_bbox(bbox):
                continue

            aoi_type = str(item.get("type", "mixed")).strip().lower()
            if aoi_type not in ALLOWED_AOI_TYPES:
                aoi_type = "mixed"

            confidence = item.get("confidence", 0.70)
            try:
                confidence_float = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_float = 0.70

            aoi_id = str(item.get("aoi_id") or f"llm_aoi_{index}")
            if not aoi_id.startswith("llm_aoi_"):
                aoi_id = f"llm_aoi_{index}"

            validated.append(
                {
                    "aoi_id": aoi_id,
                    "bbox": [float(value) for value in bbox],
                    "type": aoi_type,
                    "text": str(item.get("text", "")).strip(),
                    "source": "llm_guided",
                    "group_confidence": round(confidence_float, 3),
                    "include_in_learning": aoi_type != "footer",
                }
            )

        if not validated:
            raise ValueError("LLM AOI response contained no valid AOIs")
        return validated

    @staticmethod
    def _valid_bbox(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 4:
            return False
        try:
            x_min, y_min, x_max, y_max = [float(item) for item in value]
        except (TypeError, ValueError):
            return False
        return (
            0.0 <= x_min < x_max <= 1.0
            and 0.0 <= y_min < y_max <= 1.0
        )
