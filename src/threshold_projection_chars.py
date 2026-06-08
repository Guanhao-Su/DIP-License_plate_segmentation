"""Character segmentation with thresholding and vertical projection.

This script implements the baseline character splitter used in the course
experiment:

1. crop the plate ROI from either GT or a detector prediction
2. grayscale + CLAHE + Gaussian blur
3. build multiple foreground masks with Otsu/adaptive thresholding
4. segment characters by vertical projection
5. fall back to a 7-character prior when projection is unreliable

The output CSV follows the shared prediction interface used by evaluate.py and
visualize_single.py. All boxes are image-coordinate [x, y, w, h] boxes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


METHOD_NAME = "threshold_projection_chars_v1"
TARGET_CHAR_COUNT = 7


@dataclass(frozen=True)
class SegmenterParams:
    roi_height: int = 90
    margin_x: float = 0.03
    margin_top: float = 0.12
    margin_bottom: float = 0.10
    clahe_clip_limit: float = 2.0
    clahe_tile: tuple[int, int] = (8, 8)
    blur_kernel: int = 3
    adaptive_block_size: int = 21
    adaptive_c: int = 5
    close_kernel: tuple[int, int] = (2, 2)
    projection_smooth_window: int = 3
    min_col_foreground_frac: float = 0.10
    min_char_width_frac: float = 0.035
    max_char_width_frac: float = 0.22
    target_char_count: int = TARGET_CHAR_COUNT
    gap_merge_px: int = 1
    segment_pad_x: int = 1
    box_pad_y_frac: float = 0.04
    min_char_height_frac: float = 0.45


@dataclass
class SegmentationResult:
    boxes_resized: list[list[int]]
    mask: np.ndarray
    binary_mode: str
    strategy: str
    score: float
    raw_segment_count: int
    projection_segments: list[tuple[int, int]]


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


def clamp_bbox(box: list[int], image_shape: tuple[int, int, int]) -> list[int] | None:
    height, width = image_shape[:2]
    x, y, w, h = box
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def gt_plate_bbox(row: pd.Series) -> list[int]:
    return [
        int(row["plate_x"]),
        int(row["plate_y"]),
        int(row["plate_w"]),
        int(row["plate_h"]),
    ]


def params_json(params: SegmenterParams, plate_source: str, method_name: str, preset: str) -> str:
    values: dict[str, Any] = asdict(params)
    values["plate_source"] = plate_source
    values["method_name"] = method_name
    values["preset"] = preset
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def params_for_preset(preset: str) -> SegmenterParams:
    if preset == "base":
        return SegmenterParams()
    if preset == "box_y_pad":
        return SegmenterParams(box_pad_y_frac=0.08)
    if preset == "expanded_roi":
        return SegmenterParams(
            margin_top=0.08,
            margin_bottom=0.05,
            box_pad_y_frac=0.08,
            min_char_height_frac=0.65,
            segment_pad_x=2,
        )
    raise ValueError(f"Unknown preset: {preset}")


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
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def resolve_plate_bbox(
    row: pd.Series,
    plate_source: str,
    plate_predictions: pd.DataFrame | None,
) -> tuple[list[int] | None, str, str, float]:
    if plate_source == "gt":
        return gt_plate_bbox(row), "success", "", 0.0

    image_name = row["image_name"]
    if plate_predictions is None or image_name not in plate_predictions.index:
        return None, "plate_not_found", "missing_plate_prediction_row", 0.0

    pred_row = plate_predictions.loc[image_name]
    plate_runtime_ms = numeric_or_zero(pred_row["runtime_ms"]) if "runtime_ms" in pred_row else 0.0
    plate_box, parse_error = parse_plate_bbox(pred_row["plate_bbox_pred"])
    if parse_error:
        return None, "invalid_prediction", parse_error, plate_runtime_ms
    if plate_box is None:
        reason = ""
        if "failure_reason" in pred_row and not is_empty_value(pred_row["failure_reason"]):
            reason = str(pred_row["failure_reason"])
        return None, "plate_not_found", reason or "empty_plate_bbox", plate_runtime_ms
    return plate_box, "success", "", plate_runtime_ms


def preprocess_roi(roi: np.ndarray, params: SegmenterParams) -> tuple[np.ndarray, float, float]:
    roi_h, roi_w = roi.shape[:2]
    scale = params.roi_height / float(roi_h)
    resized_w = max(1, int(round(roi_w * scale)))
    resized = cv2.resize(roi, (resized_w, params.roi_height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip_limit, tileGridSize=params.clahe_tile)
    gray = clahe.apply(gray)
    if params.blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)
    scale_x = resized_w / float(roi_w)
    scale_y = params.roi_height / float(roi_h)
    return gray, scale_x, scale_y


def make_inner_bounds(shape: tuple[int, int], params: SegmenterParams) -> tuple[int, int, int, int]:
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


def clean_mask(mask: np.ndarray, params: SegmenterParams) -> np.ndarray:
    cleaned = mask
    if params.close_kernel[0] > 0 and params.close_kernel[1] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params.close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def build_binary_masks(gray: np.ndarray, params: SegmenterParams) -> dict[str, np.ndarray]:
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
    masks = {
        "otsu_bright": otsu_bright,
        "otsu_dark": otsu_dark,
        "adaptive_bright": adaptive_bright,
        "adaptive_dark": adaptive_dark,
    }
    return {name: clean_mask(mask, params) for name, mask in masks.items()}


def smooth_projection(projection: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return projection.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(projection.astype(np.float32), kernel, mode="same")


def bool_runs(values: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(values):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def merge_close_segments(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def extract_projection_segments(
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: SegmenterParams,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    x0, x1, y0, y1 = inner_bounds
    inner = mask[y0:y1, x0:x1]
    if inner.size == 0:
        return [], np.array([], dtype=np.float32)
    projection = np.count_nonzero(inner, axis=0) / float(inner.shape[0])
    projection = smooth_projection(projection, params.projection_smooth_window)
    active = projection >= params.min_col_foreground_frac
    raw_segments = bool_runs(active)
    raw_segments = merge_close_segments(raw_segments, params.gap_merge_px)

    inner_width = x1 - x0
    min_width = max(1, int(round(inner_width * params.min_char_width_frac)))
    max_width = max(min_width + 1, int(round(inner_width * params.max_char_width_frac)))
    filtered = []
    for start, end in raw_segments:
        width = end - start
        if width < min_width:
            continue
        if width > max_width * 2:
            filtered.append((x0 + start, x0 + end))
        elif width <= max_width:
            filtered.append((x0 + start, x0 + end))
        else:
            filtered.append((x0 + start, x0 + end))
    return filtered, projection


def local_minimum_split(interval: tuple[int, int], projection_full: np.ndarray) -> tuple[int, int] | None:
    start, end = interval
    width = end - start
    if width < 6:
        return None
    lo = start + max(2, int(round(width * 0.35)))
    hi = start + min(width - 2, int(round(width * 0.65)))
    if hi <= lo:
        return None
    local = projection_full[lo:hi]
    if local.size == 0:
        return None
    split_x = int(lo + np.argmin(local))
    if split_x <= start + 1 or split_x >= end - 1:
        return None
    return (start, split_x), (split_x, end)


def merge_to_target(segments: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    merged = sorted(segments)
    while len(merged) > target:
        best_idx = 0
        best_cost = float("inf")
        for idx in range(len(merged) - 1):
            gap = merged[idx + 1][0] - merged[idx][1]
            combined_width = merged[idx + 1][1] - merged[idx][0]
            cost = gap * 4 + combined_width
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        merged[best_idx] = (merged[best_idx][0], merged[best_idx + 1][1])
        del merged[best_idx + 1]
    return merged


def split_to_target(
    segments: list[tuple[int, int]],
    projection_full: np.ndarray,
    target: int,
) -> list[tuple[int, int]]:
    split_segments = sorted(segments)
    while len(split_segments) < target and split_segments:
        widths = [end - start for start, end in split_segments]
        widest_idx = int(np.argmax(widths))
        split_pair = local_minimum_split(split_segments[widest_idx], projection_full)
        if split_pair is None:
            break
        split_segments[widest_idx : widest_idx + 1] = list(split_pair)
    return split_segments


def prior_intervals(inner_bounds: tuple[int, int, int, int], target: int) -> list[tuple[int, int]]:
    x0, x1, _, _ = inner_bounds
    width = x1 - x0
    intervals = []
    for idx in range(target):
        start = int(round(x0 + idx * width / target))
        end = int(round(x0 + (idx + 1) * width / target))
        intervals.append((start, max(start + 1, end)))
    return intervals


def normalize_segments(
    segments: list[tuple[int, int]],
    projection: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: SegmenterParams,
) -> tuple[list[tuple[int, int]], str]:
    target = params.target_char_count
    x0, _, _, _ = inner_bounds
    if len(segments) == target:
        return segments, "projection_7"

    projection_full = np.zeros(inner_bounds[1], dtype=np.float32)
    if projection.size:
        projection_full[x0 : x0 + len(projection)] = projection

    if len(segments) > target:
        merged = merge_to_target(segments, target)
        if len(merged) == target:
            return merged, "merged_projection"

    if 0 < len(segments) < target:
        split = split_to_target(segments, projection_full, target)
        if len(split) == target:
            return split, "split_projection"

    return prior_intervals(inner_bounds, target), "prior_slots"


def box_from_interval(
    interval: tuple[int, int],
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: SegmenterParams,
) -> list[int]:
    x0, x1, y0, y1 = inner_bounds
    height, width = mask.shape[:2]
    start = max(0, min(width - 1, interval[0] - params.segment_pad_x))
    end = max(start + 1, min(width, interval[1] + params.segment_pad_x))
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

    start = max(x0, start)
    end = min(x1, end)
    return [int(start), int(top), int(max(1, end - start)), int(max(1, bottom - top))]


def boxes_from_intervals(
    intervals: list[tuple[int, int]],
    mask: np.ndarray,
    inner_bounds: tuple[int, int, int, int],
    params: SegmenterParams,
) -> list[list[int]]:
    return [box_from_interval(interval, mask, inner_bounds, params) for interval in sorted(intervals)]


def score_candidate(
    boxes: list[list[int]],
    mask: np.ndarray,
    raw_segment_count: int,
    strategy: str,
    inner_bounds: tuple[int, int, int, int],
    params: SegmenterParams,
) -> float:
    if len(boxes) != params.target_char_count:
        return -1.0

    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    mean_width = float(widths.mean()) if widths.size else 0.0
    width_balance = 0.0
    if mean_width > 0:
        width_balance = max(0.0, 1.0 - float(widths.std()) / mean_width)

    x0, x1, y0, y1 = inner_bounds
    inner_width = max(1, x1 - x0)
    span = (boxes[-1][0] + boxes[-1][2]) - boxes[0][0]
    expected_span = 0.88 * inner_width
    span_score = max(0.0, 1.0 - abs(span - expected_span) / inner_width)

    densities = []
    for bx, by, bw, bh in boxes:
        crop = mask[max(0, by) : min(mask.shape[0], by + bh), max(0, bx) : min(mask.shape[1], bx + bw)]
        area = crop.shape[0] * crop.shape[1]
        densities.append(float(np.count_nonzero(crop)) / area if area else 0.0)
    mean_density = float(np.mean(densities)) if densities else 0.0
    density_score = max(0.0, 1.0 - abs(mean_density - 0.32) / 0.32)

    count_score = max(0.0, 1.0 - abs(raw_segment_count - params.target_char_count) / params.target_char_count)
    strategy_bonus = {
        "projection_7": 0.18,
        "merged_projection": 0.10,
        "split_projection": 0.08,
        "prior_slots": 0.00,
    }.get(strategy, 0.0)
    return (
        0.32 * count_score
        + 0.28 * width_balance
        + 0.22 * density_score
        + 0.18 * span_score
        + strategy_bonus
    )


def segment_characters(gray: np.ndarray, params: SegmenterParams) -> SegmentationResult:
    masks = build_binary_masks(gray, params)
    inner_bounds = make_inner_bounds(gray.shape, params)
    candidates: list[SegmentationResult] = []

    for mode, mask in masks.items():
        raw_segments, projection = extract_projection_segments(mask, inner_bounds, params)
        normalized, strategy = normalize_segments(raw_segments, projection, inner_bounds, params)
        boxes = boxes_from_intervals(normalized, mask, inner_bounds, params)
        score = score_candidate(boxes, mask, len(raw_segments), strategy, inner_bounds, params)
        candidates.append(
            SegmentationResult(
                boxes_resized=boxes,
                mask=mask,
                binary_mode=mode,
                strategy=strategy,
                score=score,
                raw_segment_count=len(raw_segments),
                projection_segments=raw_segments,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[0]


def map_box_to_image(box: list[int], roi_box: list[int], scale_x: float, scale_y: float) -> list[int]:
    x, y, w, h = box
    mapped_x = roi_box[0] + x / scale_x
    mapped_y = roi_box[1] + y / scale_y
    mapped_w = w / scale_x
    mapped_h = h / scale_y
    return [
        int(round(mapped_x)),
        int(round(mapped_y)),
        max(1, int(round(mapped_w))),
        max(1, int(round(mapped_h))),
    ]


def run_segmentation(
    image: np.ndarray,
    plate_box: list[int],
    params: SegmenterParams,
) -> tuple[SegmentationResult | None, list[list[int]], str]:
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


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def save_binary_if_requested(
    result: SegmentationResult | None,
    plate_box: list[int] | None,
    image_name: str,
    binary_dir: Path,
    save_binary: bool,
    method_name: str,
) -> str:
    if not save_binary or result is None or plate_box is None:
        return ""
    _, _, w, h = plate_box
    binary = cv2.resize(result.mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    out_path = binary_dir / method_name / f"{Path(image_name).stem}.png"
    write_image(out_path, binary)
    return relative_or_absolute(out_path)


def make_prediction_row(
    row: pd.Series,
    image_dir: Path,
    plate_source: str,
    plate_predictions: pd.DataFrame | None,
    params: SegmenterParams,
    params_text: str,
    save_binary: bool,
    binary_dir: Path,
    method_name: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    image_name = str(row["image_name"])
    image_path = image_dir / image_name
    image = read_image(image_path)
    if image is None:
        runtime_ms = (time.perf_counter() - start) * 1000.0
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
        }

    plate_box, plate_status, plate_reason, plate_runtime_ms = resolve_plate_bbox(
        row,
        plate_source,
        plate_predictions,
    )
    if plate_box is None:
        char_runtime_ms = (time.perf_counter() - start) * 1000.0
        runtime_ms = char_runtime_ms + plate_runtime_ms
        return {
            "image_name": image_name,
            "method": method_name,
            "plate_bbox_pred": "",
            "char_bboxes_pred": "[]",
            "params": params_text,
            "runtime_ms": runtime_ms,
            "status": plate_status,
            "failure_reason": plate_reason,
            "binary_path": "",
            "plate_runtime_ms": plate_runtime_ms,
            "char_runtime_ms": char_runtime_ms,
        }

    result, char_boxes, segment_reason = run_segmentation(image, plate_box, params)
    char_runtime_ms = (time.perf_counter() - start) * 1000.0
    runtime_ms = char_runtime_ms + plate_runtime_ms
    status = "success" if result is not None and len(char_boxes) == params.target_char_count else "char_failed"
    failure_reason = segment_reason
    binary_path = save_binary_if_requested(result, plate_box, image_name, binary_dir, save_binary, method_name)

    return {
        "image_name": image_name,
        "method": method_name,
        "plate_bbox_pred": json.dumps(plate_box, ensure_ascii=False),
        "char_bboxes_pred": json.dumps(char_boxes, ensure_ascii=False),
        "params": params_text,
        "runtime_ms": runtime_ms,
        "status": status,
        "failure_reason": failure_reason,
        "binary_path": binary_path,
        "plate_runtime_ms": plate_runtime_ms,
        "char_runtime_ms": char_runtime_ms,
        "plate_source": plate_source,
        "selected_binary_mode": result.binary_mode if result is not None else "",
        "segmentation_strategy": result.strategy if result is not None else "",
        "raw_segment_count": result.raw_segment_count if result is not None else 0,
        "segmentation_score": result.score if result is not None else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="annotations/plate_char_annotations.csv", help="Ground-truth CSV.")
    parser.add_argument("--image-dir", default="dataset", help="Directory containing BMP images.")
    parser.add_argument(
        "--out",
        default="results/threshold_projection_chars_v1_predictions.csv",
        help="Output prediction CSV.",
    )
    parser.add_argument(
        "--method-name",
        default=METHOD_NAME,
        help="Method name written to the prediction CSV.",
    )
    parser.add_argument("--split", choices=["all", "tune", "test"], default="all", help="Subset to run.")
    parser.add_argument(
        "--preset",
        choices=["base", "box_y_pad", "expanded_roi"],
        default="base",
        help="Parameter preset. base preserves the original v1 behavior.",
    )
    parser.add_argument(
        "--plate-source",
        choices=["gt", "pred"],
        default="gt",
        help="Use GT plate boxes or boxes from --plate-pred.",
    )
    parser.add_argument(
        "--plate-pred",
        default="results/edge_morph_plate_v1_predictions.csv",
        help="Plate prediction CSV used when --plate-source pred.",
    )
    parser.add_argument("--save-binary", action="store_true", help="Save selected binary masks for visualization.")
    parser.add_argument(
        "--binary-dir",
        default="outputs/binary_threshold_projection_chars_v1",
        help="Output directory for selected binary masks.",
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
            args.save_binary,
            Path(args.binary_dir),
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
