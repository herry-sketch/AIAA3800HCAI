from __future__ import annotations

import math
from typing import Iterable

from .contracts import BBox, FaceLandmarks, GridName, Landmark2D

GRID_ORDER: list[GridName] = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]

GRID_ALIASES = {
    "left_top": "top_left",
    "center_top": "top_center",
    "right_top": "top_right",
    "left_middle": "middle_left",
    "center_middle": "middle_center",
    "right_middle": "middle_right",
    "left_bottom": "bottom_left",
    "center_bottom": "bottom_center",
    "right_bottom": "bottom_right",
}

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14

GAZE_FEATURE_WEIGHTS = {
    "yaw": 0.45,
    "pitch": 0.45,
    "roll": 0.15,
    "left_iris_offset_x": 1.0,
    "left_iris_offset_y": 1.0,
    "right_iris_offset_x": 1.0,
    "right_iris_offset_y": 1.0,
    "left_eye_aspect_ratio": 0.08,
    "right_eye_aspect_ratio": 0.08,
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def canonicalize_grid_name(value: str) -> str:
    return GRID_ALIASES.get(value, value)


def gaze_feature_weight(name: str) -> float:
    return GAZE_FEATURE_WEIGHTS.get(name, 0.5)


def grid_cell_bbox(grid: str) -> BBox:
    canonical = canonicalize_grid_name(grid)
    row_name, col_name = canonical.split("_")
    cols = {"left": 0, "center": 1, "right": 2}
    rows = {"top": 0, "middle": 1, "bottom": 2}
    col = cols[col_name]
    row = rows[row_name]
    cell_w = 1.0 / 3.0
    cell_h = 1.0 / 3.0
    return BBox(
        x1=col * cell_w,
        y1=row * cell_h,
        x2=(col + 1) * cell_w,
        y2=(row + 1) * cell_h,
    )


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def average_landmark(points: dict[int, Landmark2D], indices: Iterable[int]) -> Landmark2D | None:
    valid = [points[index] for index in indices if index in points]
    if not valid:
        return None
    return Landmark2D(
        x=sum(point.x for point in valid) / len(valid),
        y=sum(point.y for point in valid) / len(valid),
        z=sum(point.z for point in valid) / len(valid),
    )


def landmark_distance(points: dict[int, Landmark2D], left_idx: int, right_idx: int) -> float:
    left = points.get(left_idx)
    right = points.get(right_idx)
    if left is None or right is None:
        return 0.0
    return euclidean_distance(left.as_tuple(), right.as_tuple())


def eye_aspect_ratio(points: dict[int, Landmark2D], indices: list[int]) -> float:
    if len(indices) != 6 or any(index not in points for index in indices):
        return 0.0
    p1, p2, p3, p4, p5, p6 = [points[index] for index in indices]
    vertical_1 = euclidean_distance(p2.as_tuple(), p6.as_tuple())
    vertical_2 = euclidean_distance(p3.as_tuple(), p5.as_tuple())
    horizontal = euclidean_distance(p1.as_tuple(), p4.as_tuple())
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(points: dict[int, Landmark2D]) -> float:
    if any(index not in points for index in [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, UPPER_LIP, LOWER_LIP]):
        return 0.0
    horizontal = landmark_distance(points, MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT)
    vertical = landmark_distance(points, UPPER_LIP, LOWER_LIP)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def build_named_landmarks(face_landmarks: FaceLandmarks) -> dict[str, Landmark2D]:
    points = face_landmarks.points
    named = {
        "nose_tip": points.get(1),
        "chin": points.get(152),
        "left_eye_outer": points.get(33),
        "left_eye_inner": points.get(133),
        "right_eye_inner": points.get(362),
        "right_eye_outer": points.get(263),
        "mouth_left": points.get(61),
        "mouth_right": points.get(291),
        "left_iris": average_landmark(points, LEFT_IRIS_INDICES),
        "right_iris": average_landmark(points, RIGHT_IRIS_INDICES),
    }
    return {key: value for key, value in named.items() if value is not None}


def normalized_eye_offsets(points: dict[int, Landmark2D]) -> tuple[float, float, float, float]:
    left_center = average_landmark(points, [33, 133])
    right_center = average_landmark(points, [362, 263])
    left_iris = average_landmark(points, LEFT_IRIS_INDICES)
    right_iris = average_landmark(points, RIGHT_IRIS_INDICES)
    left_width = landmark_distance(points, 33, 133)
    right_width = landmark_distance(points, 362, 263)
    left_lids = average_landmark(points, [159, 145])
    right_lids = average_landmark(points, [386, 374])
    left_height = landmark_distance({0: points.get(159, Landmark2D(0, 0)), 1: points.get(145, Landmark2D(0, 0))}, 0, 1)
    right_height = landmark_distance({0: points.get(386, Landmark2D(0, 0)), 1: points.get(374, Landmark2D(0, 0))}, 0, 1)

    left_x = 0.0
    left_y = 0.0
    right_x = 0.0
    right_y = 0.0

    if left_center and left_iris and left_width > 0:
        left_x = (left_iris.x - left_center.x) / left_width
    if right_center and right_iris and right_width > 0:
        right_x = (right_iris.x - right_center.x) / right_width
    if left_lids and left_iris and left_height > 0:
        left_y = (left_iris.y - left_lids.y) / left_height
    if right_lids and right_iris and right_height > 0:
        right_y = (right_iris.y - right_lids.y) / right_height

    return (left_x, left_y, right_x, right_y)
