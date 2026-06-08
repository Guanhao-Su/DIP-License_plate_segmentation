from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


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
