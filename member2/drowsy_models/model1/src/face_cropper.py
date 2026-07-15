from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _looks_normalized(points: np.ndarray) -> bool:
    if points.size == 0:
        return False
    finite = points[np.isfinite(points)]
    if finite.size == 0:
        return False
    return float(np.max(np.abs(finite))) <= 2.0


def _clip_square(
    center_x: float,
    center_y: float,
    side: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    side = int(round(min(side, float(min(image_width, image_height)))))
    side = max(side, 1)
    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x1 = max(0, min(x1, image_width - side))
    y1 = max(0, min(y1, image_height - side))
    return x1, y1, x1 + side, y1 + side


def compute_face_bbox(
    landmarks: np.ndarray | Sequence[Sequence[float]] | None,
    image_width: int,
    image_height: int,
    horizontal_margin: float = 0.20,
    top_margin: float = 0.20,
    bottom_margin: float = 0.25,
    min_face_width_pixels: int = 120,
    min_face_height_pixels: int = 120,
) -> tuple[int, int, int, int] | None:
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None

    if _looks_normalized(points[:, :2]):
        xs = points[:, 0] * float(image_width)
        ys = points[:, 1] * float(image_height)
    else:
        xs = points[:, 0]
        ys = points[:, 1]

    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite_mask):
        return None

    xs = xs[finite_mask]
    ys = ys[finite_mask]
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    width = x_max - x_min
    height = y_max - y_min
    if width < min_face_width_pixels or height < min_face_height_pixels:
        return None

    x_pad = width * horizontal_margin
    y_pad_top = height * top_margin
    y_pad_bottom = height * bottom_margin

    x_min = max(0.0, x_min - x_pad)
    x_max = min(float(image_width), x_max + x_pad)
    y_min = max(0.0, y_min - y_pad_top)
    y_max = min(float(image_height), y_max + y_pad_bottom)

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    side = max(width, height)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return _clip_square(center_x, center_y, side, image_width, image_height)


def crop_face_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    output_size: int = 224,
) -> np.ndarray | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = np.asarray(image)[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return resize_image(crop, output_size, output_size)


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("Expected HxWxC image")
    src_h, src_w = image.shape[:2]
    if src_h == height and src_w == width:
        return image.copy()

    y_idx = np.linspace(0, src_h - 1, height).round().astype(int)
    x_idx = np.linspace(0, src_w - 1, width).round().astype(int)
    return image[y_idx][:, x_idx]

