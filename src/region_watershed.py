"""Member C algorithm: fused plate localization plus region/watershed chars.

The script follows the shared prediction CSV interface used by evaluate.py and
visualize_single.py. It can run in three modes:

1. --plate-source gt
   Use annotated plate boxes and evaluate only the character splitter.
2. --plate-source pred
   Use plate boxes from another prediction CSV, then run the C splitter.
3. --plate-source auto
   Detect the plate with the C fused edge/color detector, then split chars.

All output boxes use original-image [x, y, w, h] coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


METHOD_NAME = "member_c_fusion_region_watershed_v1"
TARGET_CHAR_COUNT = 7


@dataclass(frozen=True)
class PlateParams:
    clahe_clip: float = 2.0
    clahe_tile: tuple[int, int] = (8, 8)
    blur_kernel: int = 5
    sobel_kernel: int = 3
    edge_open_kernel: tuple[int, int] = (3, 3)
    edge_close_kernel: tuple[int, int] = (23, 5)
    edge_dilate_kernel: tuple[int, int] = (3, 3)
    color_open_kernel: tuple[int, int] = (3, 3)
    color_close_kernel: tuple[int, int] = (17, 5)
    min_area: int = 900
    min_width: int = 80
    max_width: int = 300
    min_height: int = 24
    max_height: int = 115
    aspect_min: float = 2.0
    aspect_max: float = 5.2
    target_aspect: float = 2.85
    target_width: int = 170
    target_height: int = 60
    box_pad_x_frac: float = 0.02
    box_pad_y_frac: float = 0.04
    nms_iou: float = 0.55
    fusion_pair_iou: float = 0.12
    score_weight_aspect: float = 0.30
    score_weight_edge_density: float = 0.25
    score_weight_color_ratio: float = 0.20
    score_weight_rectangularity: float = 0.15
    score_weight_position: float = 0.10


@dataclass(frozen=True)
class CharParams:
    roi_height: int = 96
    margin_x: float = 0.035
    margin_top: float = 0.10
    margin_bottom: float = 0.08
    clahe_clip: float = 2.0
    clahe_tile: tuple[int, int] = (8, 8)
    blur_kernel: int = 3
    adaptive_block_size: int = 21
    adaptive_c: int = 5
    seed_erode_kernel: tuple[int, int] = (2, 2)
    grow_kernel: tuple[int, int] = (3, 3)
    grow_iterations: int = 7
    grow_similarity: int = 34
    max_grow_foreground_frac: float = 0.60
    clean_open_kernel: tuple[int, int] = (2, 2)
    clean_close_kernel: tuple[int, int] = (2, 2)
    watershed_distance_ratio: float = 0.34
    watershed_min_marker_area: int = 8
    projection_smooth_window: int = 3
    min_col_foreground_frac: float = 0.08
    min_char_width_frac: float = 0.035
    max_char_width_frac: float = 0.24
    min_char_height_frac: float = 0.42
    segment_pad_x: int = 1
    box_pad_y_frac: float = 0.04
    target_char_count: int = TARGET_CHAR_COUNT


@dataclass(frozen=True)
class MemberCParams:
    plate: PlateParams = field(default_factory=PlateParams)
    chars: CharParams = field(default_factory=CharParams)


@dataclass
class PlateCandidate:
    bbox: list[int]
    score: float
    source: str
    aspect: float
    edge_density: float
    color_ratio: float
    rectangularity: float


@dataclass
class CharSegmentationResult:
    boxes_resized: list[list[int]]
    mask: np.ndarray
    marker_image: np.ndarray
    mode: str
    strategy: str
    score: float
    raw_box_count: int


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def kernel(size: tuple[int, int]) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)


def normalize_bbox(box: Any) -> list[int] | None:
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"bbox must be [x, y, w, h], got {box!r}")
    values = [int(round(float(item))) for item in box]
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values


def parse_json_value(value: Any, field_name: str) -> Any:
    if is_empty_value(value):
        return None
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be JSON text, got {type(value).__name__}")
    text = value.strip()
    if text in {"[]", "null"}:
        return None
    return json.loads(text)


def parse_plate_bbox(value: Any) -> tuple[list[int] | None, str | None]:
    try:
        return normalize_bbox(parse_json_value(value, "plate_bbox_pred")), None
    except Exception as exc:  # noqa: BLE001 - keep row-level processing robust.
        return None, str(exc)


def clamp_bbox(box: list[int], image_shape: tuple[int, ...]) -> list[int] | None:
    height, width = image_shape[:2]
    x, y, w, h = box
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def expand_bbox(box: list[int], image_shape: tuple[int, ...], pad_x_frac: float, pad_y_frac: float) -> list[int] | None:
    x, y, w, h = box
    pad_x = int(round(w * pad_x_frac))
    pad_y = int(round(h * pad_y_frac))
    return clamp_bbox([x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y], image_shape)


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    return 0.0 if union_area <= 0 else float(inter_area / union_area)


def bbox_union(box_a: list[int], box_b: list[int]) -> list[int]:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def gt_plate_bbox(row: pd.Series) -> list[int]:
    return [
        int(row["plate_x"]),
        int(row["plate_y"]),
        int(row["plate_w"]),
        int(row["plate_h"]),
    ]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def preprocess_plate_gray(image: np.ndarray, params: PlateParams) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_tile)
    gray = clahe.apply(gray)
    if params.blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)
    return gray


def build_edge_mask(gray: np.ndarray, params: PlateParams) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=params.sobel_kernel)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel(params.edge_open_kernel), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel(params.edge_close_kernel), iterations=1)
    return cv2.dilate(closed, kernel(params.edge_dilate_kernel), iterations=1)


def build_color_mask(image: np.ndarray, params: PlateParams) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 45, 45]), np.array([135, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 45, 70]), np.array([42, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([179, 85, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(blue, yellow), white)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(params.color_open_kernel), iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel(params.color_close_kernel), iterations=1)


def raw_boxes_from_mask(mask: np.ndarray, source: str) -> list[tuple[list[int], float, str]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[list[int], float, str]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(([int(x), int(y), int(w), int(h)], float(cv2.contourArea(contour)), source))
    return boxes


def score_plate_candidate(
    box: list[int],
    contour_area: float,
    source: str,
    edge_mask: np.ndarray,
    color_mask: np.ndarray,
    image_shape: tuple[int, ...],
    params: PlateParams,
) -> PlateCandidate | None:
    box = expand_bbox(box, image_shape, params.box_pad_x_frac, params.box_pad_y_frac) or box
    x, y, w, h = box
    area = w * h
    if area < params.min_area:
        return None
    if not (params.min_width <= w <= params.max_width):
        return None
    if not (params.min_height <= h <= params.max_height):
        return None

    aspect = w / h if h else 0.0
    if not (params.aspect_min <= aspect <= params.aspect_max):
        return None

    edge_roi = edge_mask[y : y + h, x : x + w]
    color_roi = color_mask[y : y + h, x : x + w]
    edge_density = float(np.count_nonzero(edge_roi)) / area if area else 0.0
    color_ratio = float(np.count_nonzero(color_roi)) / area if area else 0.0
    rectangularity = min(1.0, contour_area / area) if area else 0.0

    height, width = image_shape[:2]
    center_y_ratio = (y + h / 2.0) / float(height)
    aspect_score = max(0.0, 1.0 - abs(aspect - params.target_aspect) / 2.4)
    edge_score = min(1.0, edge_density / 0.20)
    color_score = min(1.0, color_ratio / 0.55)
    rectangularity_score = min(1.0, rectangularity / 0.55)
    position_score = 1.0 - min(1.0, abs(center_y_ratio - 0.50) / 0.55)

    source_bonus = {"fusion": 0.08, "edge": 0.02, "color": 0.02}.get(source, 0.0)
    score = (
        params.score_weight_aspect * aspect_score
        + params.score_weight_edge_density * edge_score
        + params.score_weight_color_ratio * color_score
        + params.score_weight_rectangularity * rectangularity_score
        + params.score_weight_position * position_score
        + source_bonus
    )

    if width > 0 and height > 0:
        rel_w = max(0.0, 1.0 - abs(w - params.target_width) / 170.0)
        rel_h = max(0.0, 1.0 - abs(h - params.target_height) / 85.0)
        score += 0.06 * rel_w * rel_h

    return PlateCandidate(
        bbox=[int(x), int(y), int(w), int(h)],
        score=float(score),
        source=source,
        aspect=float(aspect),
        edge_density=float(edge_density),
        color_ratio=float(color_ratio),
        rectangularity=float(rectangularity),
    )


def make_fusion_raw_boxes(
    edge_candidates: list[PlateCandidate],
    color_candidates: list[PlateCandidate],
    params: PlateParams,
) -> list[tuple[list[int], float, str]]:
    fused: list[tuple[list[int], float, str]] = []
    for edge in edge_candidates[:12]:
        for color in color_candidates[:12]:
            if bbox_iou(edge.bbox, color.bbox) < params.fusion_pair_iou:
                continue
            union = bbox_union(edge.bbox, color.bbox)
            fused.append((union, float(union[2] * union[3]), "fusion"))
    return fused


def nms_candidates(candidates: list[PlateCandidate], iou_threshold: float) -> list[PlateCandidate]:
    kept: list[PlateCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(bbox_iou(candidate.bbox, item.bbox) < iou_threshold for item in kept):
            kept.append(candidate)
    return kept


def detect_plate(image: np.ndarray, params: PlateParams) -> tuple[PlateCandidate | None, list[PlateCandidate]]:
    gray = preprocess_plate_gray(image, params)
    edge_mask = build_edge_mask(gray, params)
    color_mask = build_color_mask(image, params)

    edge_scored: list[PlateCandidate] = []
    color_scored: list[PlateCandidate] = []
    for box, area, source in raw_boxes_from_mask(edge_mask, "edge"):
        candidate = score_plate_candidate(box, area, source, edge_mask, color_mask, image.shape, params)
        if candidate is not None:
            edge_scored.append(candidate)
    for box, area, source in raw_boxes_from_mask(color_mask, "color"):
        candidate = score_plate_candidate(box, area, source, edge_mask, color_mask, image.shape, params)
        if candidate is not None:
            color_scored.append(candidate)

    edge_scored.sort(key=lambda item: item.score, reverse=True)
    color_scored.sort(key=lambda item: item.score, reverse=True)

    all_candidates = edge_scored + color_scored
    for box, area, source in make_fusion_raw_boxes(edge_scored, color_scored, params):
        candidate = score_plate_candidate(box, area, source, edge_mask, color_mask, image.shape, params)
        if candidate is not None:
            all_candidates.append(candidate)

    candidates = nms_candidates(all_candidates, params.nms_iou)
    candidates.sort(key=lambda item: item.score, reverse=True)
    return (candidates[0] if candidates else None), candidates


def preprocess_roi(roi: np.ndarray, params: CharParams) -> tuple[np.ndarray, float, float]:
    roi_h, roi_w = roi.shape[:2]
    scale = params.roi_height / float(roi_h)
    resized_w = max(1, int(round(roi_w * scale)))
    resized = cv2.resize(roi, (resized_w, params.roi_height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_tile)
    gray = clahe.apply(gray)
    if params.blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)
    return gray, resized_w / float(roi_w), params.roi_height / float(roi_h)


def make_inner_bounds(shape: tuple[int, int], params: CharParams) -> tuple[int, int, int, int]:
    height, width = shape
    x0 = int(round(width * params.margin_x))
    x1 = int(round(width * (1.0 - params.margin_x)))
    y0 = int(round(height * params.margin_top))
    y1 = int(round(height * (1.0 - params.margin_bottom)))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    return x0, x1, y0, y1


def clean_char_mask(mask: np.ndarray, params: CharParams) -> np.ndarray:
    cleaned = mask.copy()
    if params.clean_open_kernel[0] > 0 and params.clean_open_kernel[1] > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel(params.clean_open_kernel), iterations=1)
    if params.clean_close_kernel[0] > 0 and params.clean_close_kernel[1] > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel(params.clean_close_kernel), iterations=1)
    return cleaned


def build_seed_masks(gray: np.ndarray, params: CharParams) -> dict[str, np.ndarray]:
    _, otsu_bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, otsu_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    adaptive_bright = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params.adaptive_block_size,
        params.adaptive_c,
    )
    adaptive_dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.adaptive_block_size,
        params.adaptive_c,
    )
    return {
        "region_otsu_bright": clean_char_mask(otsu_bright, params),
        "region_otsu_dark": clean_char_mask(otsu_dark, params),
        "region_adaptive_bright": clean_char_mask(adaptive_bright, params),
        "region_adaptive_dark": clean_char_mask(adaptive_dark, params),
    }


def restrict_to_inner(mask: np.ndarray, inner_bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = inner_bounds
    restricted = np.zeros_like(mask)
    restricted[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return restricted


def region_grow(gray: np.ndarray, seed_mask: np.ndarray, inner_bounds: tuple[int, int, int, int], params: CharParams) -> np.ndarray:
    seed = restrict_to_inner(seed_mask, inner_bounds)
    if params.seed_erode_kernel[0] > 0 and params.seed_erode_kernel[1] > 0:
        eroded = cv2.erode(seed, kernel(params.seed_erode_kernel), iterations=1)
        if np.count_nonzero(eroded) >= params.watershed_min_marker_area:
            seed = eroded

    seed_bool = seed > 0
    if not np.any(seed_bool):
        return seed

    seed_values = gray[seed_bool]
    reference = float(np.median(seed_values))
    diff = np.abs(gray.astype(np.float32) - reference)
    growable = (diff <= params.grow_similarity) | (seed_mask > 0)
    growable = restrict_to_inner((growable.astype(np.uint8) * 255), inner_bounds) > 0

    grown = seed_bool.copy()
    grow_kernel = kernel(params.grow_kernel)
    max_area = int(round(params.max_grow_foreground_frac * np.prod(gray.shape)))
    for _ in range(params.grow_iterations):
        expanded = cv2.dilate(grown.astype(np.uint8), grow_kernel, iterations=1) > 0
        new_pixels = expanded & growable & (~grown)
        if not np.any(new_pixels):
            break
        grown |= new_pixels
        if np.count_nonzero(grown) > max_area:
            grown &= seed_mask > 0
            break

    grown_u8 = (grown.astype(np.uint8) * 255)
    return clean_char_mask(grown_u8, params)


def smooth_projection(projection: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return projection.astype(np.float32)
    filt = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(projection.astype(np.float32), filt, mode="same")


def bool_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(values):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def projection_intervals(
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    x0, x1, y0, y1 = inner_bounds
    inner = mask[y0:y1, x0:x1]
    if inner.size == 0:
        return [], np.array([], dtype=np.float32)
    projection = np.count_nonzero(inner, axis=0) / float(inner.shape[0])
    projection = smooth_projection(projection, params.projection_smooth_window)
    active = projection >= params.min_col_foreground_frac
    intervals = bool_runs(active)

    inner_width = x1 - x0
    min_width = max(1, int(round(inner_width * params.min_char_width_frac)))
    filtered = []
    for start, end in intervals:
        if end - start >= min_width:
            filtered.append((x0 + start, x0 + end))
    return filtered, projection


def prior_intervals(inner_bounds: tuple[int, int, int, int], target: int) -> list[tuple[int, int]]:
    x0, x1, _, _ = inner_bounds
    width = x1 - x0
    intervals = []
    for idx in range(target):
        start = int(round(x0 + idx * width / target))
        end = int(round(x0 + (idx + 1) * width / target))
        intervals.append((start, max(start + 1, end)))
    return intervals


def build_slot_markers(mask: np.ndarray, inner_bounds: tuple[int, int, int, int], params: CharParams) -> np.ndarray:
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask == 0] = 1
    x0, x1, y0, y1 = inner_bounds
    width = x1 - x0
    label = 2
    for idx in range(params.target_char_count):
        start = int(round(x0 + idx * width / params.target_char_count))
        end = int(round(x0 + (idx + 1) * width / params.target_char_count))
        center = (start + end) // 2
        half = max(2, (end - start) // 5)
        seed_x0 = max(x0, center - half)
        seed_x1 = min(x1, center + half + 1)
        slot = (mask[y0:y1, seed_x0:seed_x1] > 0)
        if np.count_nonzero(slot) == 0:
            continue
        markers[y0:y1, seed_x0:seed_x1][slot] = label
        label += 1
    return markers


def distance_markers(mask: np.ndarray, params: CharParams) -> np.ndarray:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return np.zeros(mask.shape, dtype=np.int32)
    sure_fg = (dist > params.watershed_distance_ratio * dist.max()).astype(np.uint8) * 255
    num_labels, markers = cv2.connectedComponents(sure_fg)
    filtered = np.zeros_like(markers, dtype=np.int32)
    next_label = 2
    for label in range(1, num_labels):
        area = int(np.count_nonzero(markers == label))
        if area >= params.watershed_min_marker_area:
            filtered[markers == label] = next_label
            next_label += 1
    filtered[mask == 0] = 1
    return filtered


def watershed_boxes(
    mask: np.ndarray,
    gray: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> tuple[list[list[int]], np.ndarray, str]:
    clean_mask = restrict_to_inner(mask, inner_bounds)
    markers = distance_markers(clean_mask, params)
    marker_count = len([label for label in np.unique(markers) if label > 1])
    strategy = "distance_watershed"
    if marker_count < max(3, params.target_char_count - 2):
        markers = build_slot_markers(clean_mask, inner_bounds, params)
        strategy = "slot_watershed"

    if len([label for label in np.unique(markers) if label > 1]) == 0:
        return [], markers, "no_marker"

    gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    gradient = cv2.addWeighted(cv2.convertScaleAbs(gradient_x), 0.5, cv2.convertScaleAbs(gradient_y), 0.5, 0)
    watershed_image = cv2.cvtColor(255 - gradient, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(watershed_image, markers.copy())

    boxes: list[list[int]] = []
    x0, x1, y0, y1 = inner_bounds
    inner_width = x1 - x0
    min_width = max(1, int(round(inner_width * params.min_char_width_frac)))
    max_width = max(min_width + 1, int(round(inner_width * params.max_char_width_frac * 1.8)))
    min_height = max(1, int(round((y1 - y0) * params.min_char_height_frac * 0.55)))

    for label in sorted(label for label in np.unique(labels) if label > 1):
        label_mask = ((labels == label) & (clean_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(label_mask) < params.watershed_min_marker_area:
            continue
        x, y, w, h = cv2.boundingRect(label_mask)
        if w < min_width or w > max_width or h < min_height:
            continue
        boxes.append([int(x), int(y), int(w), int(h)])
    boxes.sort(key=lambda item: item[0])
    return boxes, labels, strategy


def connected_component_boxes(
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> list[list[int]]:
    clean_mask = restrict_to_inner(mask, inner_bounds)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    x0, x1, y0, y1 = inner_bounds
    inner_width = x1 - x0
    min_width = max(1, int(round(inner_width * params.min_char_width_frac)))
    max_width = max(min_width + 1, int(round(inner_width * params.max_char_width_frac * 1.8)))
    min_height = max(1, int(round((y1 - y0) * params.min_char_height_frac * 0.55)))

    boxes: list[list[int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < params.watershed_min_marker_area:
            continue
        if w < min_width or w > max_width or h < min_height:
            continue
        boxes.append([int(x), int(y), int(w), int(h)])
    boxes.sort(key=lambda item: item[0])
    return boxes


def box_from_interval(
    interval: tuple[int, int],
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> list[int]:
    x0, x1, y0, y1 = inner_bounds
    height, width = mask.shape[:2]
    start = max(0, min(width - 1, interval[0] - params.segment_pad_x))
    end = max(start + 1, min(width, interval[1] + params.segment_pad_x))
    start = max(x0, start)
    end = min(x1, end)

    char_mask = mask[y0:y1, start:end]
    rows = np.where(np.count_nonzero(char_mask, axis=1) > 0)[0] if char_mask.size else np.array([])
    if rows.size:
        top = int(y0 + rows.min())
        bottom = int(y0 + rows.max() + 1)
        pad_y = int(round(height * params.box_pad_y_frac))
        top = max(0, top - pad_y)
        bottom = min(height, bottom + pad_y)
        if bottom - top < int(round((y1 - y0) * params.min_char_height_frac)):
            top, bottom = y0, y1
    else:
        top, bottom = y0, y1
    return [int(start), int(top), int(max(1, end - start)), int(max(1, bottom - top))]


def boxes_from_intervals(
    intervals: list[tuple[int, int]],
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> list[list[int]]:
    return [box_from_interval(interval, mask, inner_bounds, params) for interval in sorted(intervals)]


def merge_boxes_to_target(boxes: list[list[int]], target: int) -> list[list[int]]:
    merged = sorted(boxes, key=lambda item: item[0])
    while len(merged) > target:
        best_idx = 0
        best_cost = float("inf")
        for idx in range(len(merged) - 1):
            gap = merged[idx + 1][0] - (merged[idx][0] + merged[idx][2])
            combined_width = (merged[idx + 1][0] + merged[idx + 1][2]) - merged[idx][0]
            cost = gap * 4 + combined_width
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        a = merged[best_idx]
        b = merged[best_idx + 1]
        x1 = min(a[0], b[0])
        y1 = min(a[1], b[1])
        x2 = max(a[0] + a[2], b[0] + b[2])
        y2 = max(a[1] + a[3], b[1] + b[3])
        merged[best_idx] = [x1, y1, x2 - x1, y2 - y1]
        del merged[best_idx + 1]
    return merged


def split_interval_by_projection(interval: tuple[int, int], projection_full: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    start, end = interval
    width = end - start
    if width < 8:
        return None
    lo = start + max(2, int(round(width * 0.35)))
    hi = start + min(width - 2, int(round(width * 0.65)))
    if hi <= lo:
        return None
    split_x = int(lo + np.argmin(projection_full[lo:hi]))
    if split_x <= start + 1 or split_x >= end - 1:
        return None
    return (start, split_x), (split_x, end)


def normalize_intervals_to_target(
    intervals: list[tuple[int, int]],
    projection: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> tuple[list[tuple[int, int]], str]:
    target = params.target_char_count
    sorted_intervals = sorted(intervals)
    if len(sorted_intervals) == target:
        return sorted_intervals, "projection_exact"

    while len(sorted_intervals) > target:
        best_idx = 0
        best_gap = float("inf")
        for idx in range(len(sorted_intervals) - 1):
            gap = sorted_intervals[idx + 1][0] - sorted_intervals[idx][1]
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        sorted_intervals[best_idx] = (sorted_intervals[best_idx][0], sorted_intervals[best_idx + 1][1])
        del sorted_intervals[best_idx + 1]
    if len(sorted_intervals) == target:
        return sorted_intervals, "projection_merged"

    x0, x1, _, _ = inner_bounds
    projection_full = np.zeros(max(x1, x0 + len(projection)), dtype=np.float32)
    if projection.size:
        projection_full[x0 : x0 + len(projection)] = projection
    while 0 < len(sorted_intervals) < target:
        widths = [end - start for start, end in sorted_intervals]
        widest_idx = int(np.argmax(widths))
        split = split_interval_by_projection(sorted_intervals[widest_idx], projection_full)
        if split is None:
            break
        sorted_intervals[widest_idx : widest_idx + 1] = list(split)
    if len(sorted_intervals) == target:
        return sorted_intervals, "projection_split"

    return prior_intervals(inner_bounds, target), "prior_slots"


def normalize_boxes_to_target(
    boxes: list[list[int]],
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> tuple[list[list[int]], str]:
    target = params.target_char_count
    boxes = sorted(boxes, key=lambda item: item[0])
    if len(boxes) == target:
        return boxes, "box_exact"
    if len(boxes) > target:
        merged = merge_boxes_to_target(boxes, target)
        if len(merged) == target:
            return merged, "box_merged"

    intervals, projection = projection_intervals(mask, inner_bounds, params)
    normalized_intervals, strategy = normalize_intervals_to_target(intervals, projection, inner_bounds, params)
    return boxes_from_intervals(normalized_intervals, mask, inner_bounds, params), strategy


def score_char_result(
    boxes: list[list[int]],
    mask: np.ndarray,
    raw_count: int,
    strategy: str,
    inner_bounds: tuple[int, int, int, int],
    params: CharParams,
) -> float:
    if len(boxes) != params.target_char_count:
        return -1.0

    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    heights = np.array([box[3] for box in boxes], dtype=np.float32)
    mean_width = float(widths.mean()) if widths.size else 0.0
    width_balance = 0.0 if mean_width <= 0 else max(0.0, 1.0 - float(widths.std()) / mean_width)

    x0, x1, y0, y1 = inner_bounds
    inner_width = max(1, x1 - x0)
    inner_height = max(1, y1 - y0)
    span = (boxes[-1][0] + boxes[-1][2]) - boxes[0][0]
    span_score = max(0.0, 1.0 - abs(span - 0.88 * inner_width) / inner_width)
    height_score = max(0.0, 1.0 - abs(float(heights.mean()) - 0.72 * inner_height) / inner_height)

    densities = []
    for x, y, w, h in boxes:
        crop = mask[max(0, y) : min(mask.shape[0], y + h), max(0, x) : min(mask.shape[1], x + w)]
        area = crop.shape[0] * crop.shape[1]
        densities.append(float(np.count_nonzero(crop)) / area if area else 0.0)
    mean_density = float(np.mean(densities)) if densities else 0.0
    density_score = max(0.0, 1.0 - abs(mean_density - 0.30) / 0.34)
    raw_count_score = max(0.0, 1.0 - abs(raw_count - params.target_char_count) / params.target_char_count)
    strategy_bonus = {
        "distance_watershed:box_exact": 0.22,
        "slot_watershed:box_exact": 0.18,
        "distance_watershed:box_merged": 0.12,
        "slot_watershed:box_merged": 0.10,
        "component:box_exact": 0.10,
        "projection_exact": 0.08,
        "projection_merged": 0.04,
        "projection_split": 0.04,
        "prior_slots": 0.00,
    }.get(strategy, 0.0)

    return (
        0.26 * width_balance
        + 0.22 * span_score
        + 0.20 * density_score
        + 0.16 * height_score
        + 0.16 * raw_count_score
        + strategy_bonus
    )


def segment_characters(gray: np.ndarray, params: CharParams) -> CharSegmentationResult:
    inner_bounds = make_inner_bounds(gray.shape, params)
    seed_masks = build_seed_masks(gray, params)
    candidates: list[CharSegmentationResult] = []

    for mode, seed_mask in seed_masks.items():
        grown = region_grow(gray, seed_mask, inner_bounds, params)
        ws_boxes, marker_image, ws_strategy = watershed_boxes(grown, gray, inner_bounds, params)
        normalized_boxes, normalize_strategy = normalize_boxes_to_target(ws_boxes, grown, inner_bounds, params)
        strategy = f"{ws_strategy}:{normalize_strategy}" if normalize_strategy.startswith("box") else normalize_strategy
        score = score_char_result(normalized_boxes, grown, len(ws_boxes), strategy, inner_bounds, params)
        candidates.append(
            CharSegmentationResult(
                boxes_resized=normalized_boxes,
                mask=grown,
                marker_image=marker_image,
                mode=mode,
                strategy=strategy,
                score=score,
                raw_box_count=len(ws_boxes),
            )
        )

        cc_boxes = connected_component_boxes(grown, inner_bounds, params)
        normalized_cc, cc_strategy = normalize_boxes_to_target(cc_boxes, grown, inner_bounds, params)
        cc_strategy_full = f"component:{cc_strategy}" if cc_strategy.startswith("box") else cc_strategy
        cc_score = score_char_result(normalized_cc, grown, len(cc_boxes), cc_strategy_full, inner_bounds, params)
        candidates.append(
            CharSegmentationResult(
                boxes_resized=normalized_cc,
                mask=grown,
                marker_image=marker_image,
                mode=f"{mode}_cc",
                strategy=cc_strategy_full,
                score=cc_score,
                raw_box_count=len(cc_boxes),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[0]


def map_box_to_image(box: list[int], roi_box: list[int], scale_x: float, scale_y: float) -> list[int]:
    x, y, w, h = box
    return [
        int(round(roi_box[0] + x / scale_x)),
        int(round(roi_box[1] + y / scale_y)),
        max(1, int(round(w / scale_x))),
        max(1, int(round(h / scale_y))),
    ]


def run_char_segmentation(
    image: np.ndarray,
    plate_box: list[int],
    params: CharParams,
) -> tuple[CharSegmentationResult | None, list[list[int]], str]:
    roi_box = clamp_bbox(plate_box, image.shape)
    if roi_box is None:
        return None, [], "plate_bbox_outside_image"
    x, y, w, h = roi_box
    roi = image[y : y + h, x : x + w]
    if roi.size == 0:
        return None, [], "empty_plate_roi"
    gray, scale_x, scale_y = preprocess_roi(roi, params)
    result = segment_characters(gray, params)
    mapped_boxes = [map_box_to_image(box, roi_box, scale_x, scale_y) for box in result.boxes_resized]
    return result, mapped_boxes, ""


def save_debug_if_requested(
    result: CharSegmentationResult | None,
    plate_box: list[int] | None,
    image_name: str,
    debug_dir: Path,
    save_debug: bool,
    method_name: str,
) -> tuple[str, str]:
    if not save_debug or result is None or plate_box is None:
        return "", ""
    _, _, w, h = plate_box
    binary = cv2.resize(result.mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    marker = result.marker_image
    marker_vis = np.zeros((*marker.shape[:2], 3), dtype=np.uint8)
    labels = [label for label in np.unique(marker) if label > 1]
    for idx, label in enumerate(labels):
        color = (
            int((53 * idx + 70) % 255),
            int((97 * idx + 130) % 255),
            int((193 * idx + 40) % 255),
        )
        marker_vis[marker == label] = color
    marker_vis[marker == -1] = (0, 0, 255)
    marker_vis = cv2.resize(marker_vis, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)

    binary_path = debug_dir / method_name / "binary" / f"{Path(image_name).stem}.png"
    marker_path = debug_dir / method_name / "watershed" / f"{Path(image_name).stem}.png"
    write_image(binary_path, binary)
    write_image(marker_path, marker_vis)
    return relative_or_absolute(binary_path), relative_or_absolute(marker_path)


def load_ground_truth(path: Path, split: str) -> pd.DataFrame:
    gt = pd.read_csv(path, encoding="utf-8-sig")
    required = {"image_name", "split", "plate_x", "plate_y", "plate_w", "plate_h"}
    missing = required - set(gt.columns)
    if missing:
        raise SystemExit(f"Ground-truth CSV is missing columns: {sorted(missing)}")
    if split != "all":
        gt = gt[gt["split"] == split].copy()
    if gt.empty:
        raise SystemExit(f"No rows found for split={split}")
    return gt


def load_plate_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, encoding="utf-8-sig")
    required = {"image_name", "plate_bbox_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise SystemExit(f"Plate prediction CSV is missing columns: {sorted(missing)}")
    if pred["image_name"].duplicated().any():
        duplicated = pred.loc[pred["image_name"].duplicated(), "image_name"].tolist()
        raise SystemExit(f"Plate prediction CSV has duplicate image_name rows: {duplicated}")
    return pred.set_index("image_name")


def numeric_or_zero(value: Any) -> float:
    if is_empty_value(value):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) or math.isinf(number) else number


def params_json(params: MemberCParams, plate_source: str, method_name: str, preset: str) -> str:
    values: dict[str, Any] = asdict(params)
    values["plate_source"] = plate_source
    values["method_name"] = method_name
    values["preset"] = preset
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def params_for_preset(preset: str) -> MemberCParams:
    if preset == "base":
        return MemberCParams()
    if preset == "region_loose":
        return MemberCParams(
            chars=CharParams(
                grow_similarity=42,
                grow_iterations=9,
                max_grow_foreground_frac=0.66,
                watershed_distance_ratio=0.30,
            )
        )
    if preset == "plate_strict":
        return MemberCParams(
            plate=PlateParams(
                min_area=1200,
                aspect_min=2.15,
                aspect_max=4.7,
                score_weight_aspect=0.34,
                score_weight_edge_density=0.24,
                score_weight_color_ratio=0.18,
                score_weight_rectangularity=0.16,
                score_weight_position=0.08,
            )
        )
    raise ValueError(f"Unknown preset: {preset}")


def resolve_plate_bbox(
    row: pd.Series,
    image: np.ndarray,
    plate_source: str,
    plate_predictions: pd.DataFrame | None,
    params: PlateParams,
) -> tuple[list[int] | None, str, str, float, dict[str, Any]]:
    if plate_source == "gt":
        return gt_plate_bbox(row), "success", "", 0.0, {}

    if plate_source == "pred":
        image_name = str(row["image_name"])
        if plate_predictions is None or image_name not in plate_predictions.index:
            return None, "plate_not_found", "missing_plate_prediction_row", 0.0, {}
        pred_row = plate_predictions.loc[image_name]
        runtime_ms = numeric_or_zero(pred_row["runtime_ms"]) if "runtime_ms" in pred_row else 0.0
        plate_box, parse_error = parse_plate_bbox(pred_row["plate_bbox_pred"])
        if parse_error:
            return None, "invalid_prediction", parse_error, runtime_ms, {}
        if plate_box is None:
            reason = str(pred_row["failure_reason"]) if "failure_reason" in pred_row and not is_empty_value(pred_row["failure_reason"]) else ""
            return None, "plate_not_found", reason or "empty_plate_bbox", runtime_ms, {}
        return plate_box, "success", "", runtime_ms, {}

    start = time.perf_counter()
    best, candidates = detect_plate(image, params)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if best is None:
        return None, "plate_not_found", "no_fused_candidate_after_filtering", runtime_ms, {
            "plate_candidate_count": 0,
        }
    return best.bbox, "success", "", runtime_ms, {
        "plate_candidate_count": len(candidates),
        "plate_candidate_score": best.score,
        "plate_candidate_source": best.source,
        "plate_candidate_aspect": best.aspect,
        "plate_candidate_edge_density": best.edge_density,
        "plate_candidate_color_ratio": best.color_ratio,
        "plate_candidate_rectangularity": best.rectangularity,
    }


def make_prediction_row(
    row: pd.Series,
    image_dir: Path,
    plate_source: str,
    plate_predictions: pd.DataFrame | None,
    params: MemberCParams,
    params_text: str,
    save_debug: bool,
    debug_dir: Path,
    method_name: str,
) -> dict[str, Any]:
    row_start = time.perf_counter()
    image_name = str(row["image_name"])
    image_path = image_dir / image_name
    image = read_image(image_path)
    if image is None:
        runtime_ms = (time.perf_counter() - row_start) * 1000.0
        return {
            "image_name": image_name,
            "method": method_name,
            "plate_bbox_pred": "",
            "char_bboxes_pred": "[]",
            "params": params_text,
            "runtime_ms": runtime_ms,
            "status": "exception",
            "failure_reason": f"image_read_failed:{image_path.as_posix()}",
            "binary_path": "",
            "foreground_path": "",
        }

    plate_box, plate_status, plate_reason, plate_runtime_ms, plate_debug = resolve_plate_bbox(
        row,
        image,
        plate_source,
        plate_predictions,
        params.plate,
    )
    if plate_box is None:
        char_runtime_ms = (time.perf_counter() - row_start) * 1000.0
        result = {
            "image_name": image_name,
            "method": method_name,
            "plate_bbox_pred": "",
            "char_bboxes_pred": "[]",
            "params": params_text,
            "runtime_ms": plate_runtime_ms + char_runtime_ms,
            "status": plate_status,
            "failure_reason": plate_reason,
            "binary_path": "",
            "foreground_path": "",
            "plate_runtime_ms": plate_runtime_ms,
            "char_runtime_ms": char_runtime_ms,
            "plate_source": plate_source,
        }
        result.update(plate_debug)
        return result

    char_start = time.perf_counter()
    char_result, char_boxes, char_reason = run_char_segmentation(image, plate_box, params.chars)
    char_runtime_ms = (time.perf_counter() - char_start) * 1000.0
    runtime_ms = plate_runtime_ms + char_runtime_ms
    status = "success" if char_result is not None and len(char_boxes) == params.chars.target_char_count else "char_failed"
    binary_path, marker_path = save_debug_if_requested(
        char_result,
        plate_box,
        image_name,
        debug_dir,
        save_debug,
        method_name,
    )

    result = {
        "image_name": image_name,
        "method": method_name,
        "plate_bbox_pred": json.dumps(plate_box, ensure_ascii=False),
        "char_bboxes_pred": json.dumps(char_boxes, ensure_ascii=False),
        "params": params_text,
        "runtime_ms": runtime_ms,
        "status": status,
        "failure_reason": char_reason,
        "binary_path": binary_path,
        "foreground_path": marker_path,
        "plate_runtime_ms": plate_runtime_ms,
        "char_runtime_ms": char_runtime_ms,
        "plate_source": plate_source,
        "selected_region_mode": char_result.mode if char_result is not None else "",
        "segmentation_strategy": char_result.strategy if char_result is not None else "",
        "raw_region_box_count": char_result.raw_box_count if char_result is not None else 0,
        "segmentation_score": char_result.score if char_result is not None else math.nan,
    }
    result.update(plate_debug)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="annotations/plate_char_annotations.csv", help="Ground-truth CSV.")
    parser.add_argument("--image-dir", default="dataset", help="Directory containing BMP images.")
    parser.add_argument(
        "--out",
        default="results/member_c_fusion_region_watershed_v1_predictions.csv",
        help="Output prediction CSV.",
    )
    parser.add_argument("--method-name", default=METHOD_NAME, help="Method name written to the prediction CSV.")
    parser.add_argument("--split", choices=["all", "tune", "test"], default="all", help="Subset to run.")
    parser.add_argument(
        "--preset",
        choices=["base", "region_loose", "plate_strict"],
        default="base",
        help="Parameter preset.",
    )
    parser.add_argument(
        "--plate-source",
        choices=["gt", "pred", "auto"],
        default="auto",
        help="Use GT plate boxes, external predictions, or this script's fused detector.",
    )
    parser.add_argument(
        "--plate-pred",
        default="results/edge_morph_plate_rect_heavy_v2_predictions.csv",
        help="Plate prediction CSV used when --plate-source pred.",
    )
    parser.add_argument("--save-debug", action="store_true", help="Save selected binary and watershed marker images.")
    parser.add_argument(
        "--debug-dir",
        default="outputs/debug_member_c_region_watershed",
        help="Output directory for debug masks.",
    )
    args = parser.parse_args()

    params = params_for_preset(args.preset)
    params_text = params_json(params, args.plate_source, args.method_name, args.preset)
    gt = load_ground_truth(Path(args.gt), args.split)
    plate_predictions = None
    if args.plate_source == "pred":
        plate_predictions = load_plate_predictions(Path(args.plate_pred))

    image_dir = Path(args.image_dir)
    rows = [
        make_prediction_row(
            row,
            image_dir,
            args.plate_source,
            plate_predictions,
            params,
            params_text,
            args.save_debug,
            Path(args.debug_dir),
            args.method_name,
        )
        for _, row in gt.iterrows()
    ]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    success_count = sum(row["status"] == "success" for row in rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    print(f"status: success={success_count}, failed={len(rows) - success_count}")


if __name__ == "__main__":
    main()
