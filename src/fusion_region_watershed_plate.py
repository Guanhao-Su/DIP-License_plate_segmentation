from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fusion_region_watershed_common import bbox_iou, bbox_union, expand_bbox, kernel


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


@dataclass
class PlateCandidate:
    bbox: list[int]
    score: float
    source: str
    aspect: float
    edge_density: float
    color_ratio: float
    rectangularity: float


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
