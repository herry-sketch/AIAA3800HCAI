from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import ClassificationObservation


class YoloClassifierError(RuntimeError):
    """Raised when the YOLO classification model cannot be loaded."""


MIN_VALID_YOLO_WEIGHT_BYTES = 10_000_000


def _normalize_class_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def _import_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def resolve_device_config(
    requested_device: int | str | None = None,
    requested_half: bool | None = None,
) -> tuple[int | str, bool]:
    torch = _import_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if cuda_available:
        return (0 if requested_device is None else requested_device), (True if requested_half is None else requested_half)
    return "cpu", False


def resolve_yolo_weight(
    local_path: str,
    repo_id: str,
    filename: str,
) -> Path:
    local = Path(local_path).expanduser()

    if not local.is_absolute():
        local = Path.cwd() / local

    if local.exists():
        size = local.stat().st_size
        if size > MIN_VALID_YOLO_WEIGHT_BYTES:
            resolved = local.resolve()
            print("YOLO weight source: local")
            print(f"YOLO weight path: {resolved}")
            return resolved

        raise RuntimeError(
            f"Local YOLO weight is too small or invalid: "
            f"{local} ({size} bytes)"
        )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Local YOLO weight not found at {local}, "
            f"and Hugging Face download failed"
        ) from exc

    downloaded_path = Path(downloaded).expanduser().resolve()
    print("YOLO weight source: huggingface")
    print(f"YOLO weight path: {downloaded_path}")
    return downloaded_path


@dataclass
class YoloClassifier:
    local_path: str
    repo_id: str
    filename: str
    imgsz: int = 224
    device: int | str | None = None
    half: bool | None = None

    def __post_init__(self) -> None:
        self.device, self.half = resolve_device_config(self.device, self.half)
        self._model = self._load_model()
        self.class_names = self._extract_class_names()

    def _load_model(self):  # pragma: no cover - runtime dependency
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise YoloClassifierError("ultralytics is required for YOLO classification") from exc

        try:
            weight_path = resolve_yolo_weight(
                local_path=self.local_path,
                repo_id=self.repo_id,
                filename=self.filename,
            )
        except RuntimeError as exc:
            raise YoloClassifierError(str(exc)) from exc

        try:
            return YOLO(str(weight_path))
        except Exception as exc:
            raise YoloClassifierError(f"Failed to load YOLO weights: {weight_path}") from exc

    def _extract_class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            class_names = {int(index): str(name) for index, name in names.items()}
        elif isinstance(names, list):
            class_names = {index: str(name) for index, name in enumerate(names)}
        else:
            raise YoloClassifierError(f"Unexpected model.names format: {type(names)!r}")

        normalized = {_normalize_class_name(name) for name in class_names.values()}
        if "drowsy" not in normalized or "non drowsy" not in normalized:
            raise YoloClassifierError(f"Unexpected class names: {class_names}")
        print(f"YOLO classes: {class_names}")
        return class_names

    def predict(self, face_roi: np.ndarray, timestamp: float) -> ClassificationObservation:
        started = time.perf_counter()
        try:  # pragma: no cover - runtime dependency
            predict_kwargs = {
                "source": face_roi,
                "imgsz": self.imgsz,
                "device": self.device,
                "verbose": False,
            }
            if self.half and self.device != "cpu":
                predict_kwargs["quantize"] = 16
            results = self._model.predict(
                **predict_kwargs,
            )
            result = results[0]
            probs = result.probs.data.detach().float().cpu().numpy()
            names = result.names
        except Exception:
            return ClassificationObservation(
                timestamp=timestamp,
                p_drowsy=0.0,
                p_awake=0.0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                valid=False,
            )

        if isinstance(names, dict):
            class_names = {int(index): str(name) for index, name in names.items()}
        else:
            class_names = {index: str(name) for index, name in enumerate(names)}

        probabilities: dict[str, float] = {}
        for index, probability in enumerate(probs):
            probabilities[_normalize_class_name(class_names[index])] = float(probability)

        p_drowsy = probabilities.get("drowsy")
        p_awake = probabilities.get("non drowsy")
        latency_ms = (time.perf_counter() - started) * 1000.0
        if p_drowsy is None or p_awake is None:
            return ClassificationObservation(
                timestamp=timestamp,
                p_drowsy=0.0,
                p_awake=0.0,
                latency_ms=latency_ms,
                valid=False,
            )

        return ClassificationObservation(
            timestamp=timestamp,
            p_drowsy=float(p_drowsy),
            p_awake=float(p_awake),
            latency_ms=latency_ms,
            valid=True,
        )
