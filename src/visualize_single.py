"""Create 2x2 single-image visualizations for one or more prediction methods.

Each output image contains:

1. Original image with ground-truth and predicted plate boxes.
2. Plate ROI, cropped from predicted plate if available, otherwise from GT.
3. Plate ROI with only ground-truth character boxes.
4. Plate ROI with only predicted character boxes.

Prediction boxes must use original-image coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


EXPECTED_CHAR_COUNT = 7
REQUIRED_GT_COLUMNS = {
    "image_name",
    "plate_x",
    "plate_y",
    "plate_w",
    "plate_h",
    "char_bboxes_gt",
}
REQUIRED_PRED_COLUMNS = {
    "image_name",
    "method",
    "plate_bbox_pred",
    "char_bboxes_pred",
}


GREEN = (60, 220, 80)
RED = (40, 40, 240)
YELLOW = (40, 220, 240)
CYAN = (240, 210, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)


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


def parse_json_value(value: Any) -> Any:
    if is_empty_value(value):
        return None
    if isinstance(value, (list, tuple)):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Expected JSON text, got {type(value).__name__}")
    text = value.strip()
    if text in {"[]", "null"}:
        return None
    return json.loads(text)


def normalize_bbox(box: Any) -> list[int] | None:
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"bbox must be [x, y, w, h], got {box!r}")
    values = [int(round(float(item))) for item in box]
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values


def parse_plate_bbox(value: Any) -> tuple[list[int] | None, str | None]:
    try:
        return normalize_bbox(parse_json_value(value)), None
    except Exception as exc:  # noqa: BLE001 - keep visualization robust.
        return None, str(exc)


def parse_char_bboxes(value: Any) -> tuple[list[list[int]], str | None]:
    try:
        parsed = parse_json_value(value)
        if parsed is None:
            return [], None
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"char boxes must be a list, got {parsed!r}")
        boxes = []
        for box in parsed:
            normalized = normalize_bbox(box)
            if normalized is not None:
                boxes.append(normalized)
        boxes.sort(key=lambda item: item[0])
        return boxes, None
    except Exception as exc:  # noqa: BLE001 - keep visualization robust.
        return [], str(exc)


def gt_plate_bbox(row: pd.Series) -> list[int]:
    return [
        int(row["plate_x"]),
        int(row["plate_y"]),
        int(row["plate_w"]),
        int(row["plate_h"]),
    ]


def gt_char_bboxes(row: pd.Series) -> list[list[int]]:
    boxes = json.loads(row["char_bboxes_gt"])
    return [normalize_bbox(box) for box in boxes if normalize_bbox(box) is not None]


def clamp_bbox(box: list[int], image_shape: tuple[int, int, int]) -> list[int] | None:
    height, width = image_shape[:2]
    x, y, w, h = box
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def crop_image(image: np.ndarray, box: list[int] | None) -> np.ndarray:
    clamped = clamp_bbox(box, image.shape) if box is not None else None
    if clamped is None:
        return make_placeholder(320, 160, "No ROI")
    x, y, w, h = clamped
    return image[y : y + h, x : x + w].copy()


def bbox_to_local(box: list[int], roi_box: list[int]) -> list[int]:
    return [box[0] - roi_box[0], box[1] - roi_box[1], box[2], box[3]]


def draw_bbox(
    image: np.ndarray,
    box: list[int] | None,
    color: tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    if box is None:
        return
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if label:
        draw_label(image, label, (x, max(18, y - 4)), color)


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(image.shape[1] - tw - 2, x))
    y = max(th + 4, min(image.shape[0] - 2, y))
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), BLACK, -1)
    cv2.putText(image, text, (x + 2, y - 3), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), BLACK, -1)
    cv2.putText(
        image,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    titled = image.copy()
    draw_panel_title(titled, title)
    return titled


def draw_centered_label(
    image: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(round(center_x - tw / 2))
    x = max(2, min(image.shape[1] - tw - 2, x))
    y = max(th + 3, min(image.shape[0] - baseline - 2, y))
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bbox_plain(
    image: np.ndarray,
    box: list[int] | None,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if box is None:
        return
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def make_char_panel(
    roi: np.ndarray,
    boxes: list[list[int]],
    roi_box: list[int] | None,
    title: str,
    label_prefix: str,
    color: tuple[int, int, int],
    empty_text: str,
    thickness: int,
) -> np.ndarray:
    label_h = 40
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if roi.size == 0:
        roi = make_placeholder(320, 160, "Empty ROI")
    panel = np.full((roi.shape[0] + label_h, roi.shape[1], 3), 245, dtype=np.uint8)
    panel[label_h:, :] = roi
    cv2.rectangle(panel, (0, 0), (panel.shape[1], label_h), BLACK, -1)
    cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA)

    if roi_box is not None:
        for idx, box in enumerate(boxes, start=1):
            local = bbox_to_local(box, roi_box)
            draw_box = [local[0], local[1] + label_h, local[2], local[3]]
            draw_bbox_plain(panel, draw_box, color, thickness)
            center_x = int(round(local[0] + local[2] / 2))
            draw_centered_label(panel, f"{label_prefix}{idx}", center_x, 35, color)

    if not boxes:
        draw_label(panel, empty_text, (8, label_h + 28), RED)
    return panel


def resize_to_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    panel_w, panel_h = size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return make_placeholder(panel_w, panel_h, "Empty image")
    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_placeholder(width: int, height: int, text: str) -> np.ndarray:
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(image, text, (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    return image


def safe_filename(text: str) -> str:
    return re.sub(r'[<>:"/\\\\|?*]+', "_", text)


def build_metrics_lookup(metrics_path: Path | None) -> pd.DataFrame | None:
    if metrics_path is None:
        return None
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file does not exist: {metrics_path}")
    metrics = pd.read_csv(metrics_path, encoding="utf-8-sig")
    required = {"image_name", "method", "plate_iou", "char_count_pred", "end_to_end_success"}
    missing = required - set(metrics.columns)
    if missing:
        raise SystemExit(f"Metrics CSV is missing columns: {sorted(missing)}")
    return metrics


def metrics_text(row: pd.Series, metrics_row: pd.Series | None, parse_errors: list[str]) -> list[str]:
    lines = [f"method={row['method']}"]
    status = row["status"] if "status" in row and not is_empty_value(row["status"]) else ""
    if status:
        lines.append(f"status={status}")
    if metrics_row is not None:
        lines.append(f"IoU={float(metrics_row['plate_iou']):.3f}")
        lines.append(f"chars={int(metrics_row['char_count_pred'])}/7")
        ok = bool(metrics_row["end_to_end_success"])
        lines.append(f"E2E={'OK' if ok else 'FAIL'}")
    if parse_errors:
        lines.append("parse_error")
    return lines


def render_visualization(
    image: np.ndarray,
    gt_row: pd.Series,
    pred_row: pd.Series,
    metrics_row: pd.Series | None,
    project_root: Path,
) -> np.ndarray:
    gt_plate = gt_plate_bbox(gt_row)
    gt_chars = gt_char_bboxes(gt_row)
    pred_plate, plate_error = parse_plate_bbox(pred_row["plate_bbox_pred"])
    pred_chars, chars_error = parse_char_bboxes(pred_row["char_bboxes_pred"])
    parse_errors = [error for error in (plate_error, chars_error) if error]
    roi_box = pred_plate or gt_plate
    clamped_roi = clamp_bbox(roi_box, image.shape) or clamp_bbox(gt_plate, image.shape)
    roi = crop_image(image, clamped_roi)

    original_panel = image.copy()
    draw_bbox(original_panel, gt_plate, GREEN, "GT plate", 2)
    draw_bbox(original_panel, pred_plate, RED, "Pred plate", 2)
    for idx, line in enumerate(metrics_text(pred_row, metrics_row, parse_errors)):
        draw_label(original_panel, line, (8, 28 + idx * 22), RED if "FAIL" in line or "error" in line else WHITE)

    roi_panel = roi.copy()
    panel_title = "Plate ROI (pred)" if pred_plate is not None else "Plate ROI (GT fallback)"

    gt_chars_panel = make_char_panel(
        roi,
        gt_chars,
        clamped_roi,
        f"GT chars: {len(gt_chars)}/{EXPECTED_CHAR_COUNT}",
        "G",
        GREEN,
        "No GT chars",
        1,
    )
    pred_chars_panel = make_char_panel(
        roi,
        pred_chars,
        clamped_roi,
        f"Pred chars: {len(pred_chars)}/{EXPECTED_CHAR_COUNT}",
        "P",
        YELLOW,
        "No predicted chars",
        2,
    )

    panel_size = (520, 360)
    panels = [
        resize_to_panel(original_panel, panel_size),
        add_panel_title(resize_to_panel(roi_panel, panel_size), panel_title),
        resize_to_panel(gt_chars_panel, panel_size),
        resize_to_panel(pred_chars_panel, panel_size),
    ]
    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    canvas = np.vstack([top, bottom])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), GRAY, 1)
    return canvas


def load_inputs(gt_path: Path, pred_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    missing_gt = REQUIRED_GT_COLUMNS - set(gt.columns)
    missing_pred = REQUIRED_PRED_COLUMNS - set(pred.columns)
    if missing_gt:
        raise SystemExit(f"GT CSV is missing columns: {sorted(missing_gt)}")
    if missing_pred:
        raise SystemExit(f"Prediction CSV is missing columns: {sorted(missing_pred)}")
    for column in ("status", "binary_path", "foreground_path"):
        if column not in pred.columns:
            pred[column] = ""
    return gt, pred


def apply_filters(
    pred: pd.DataFrame,
    metrics: pd.DataFrame | None,
    method: str | None,
    only_failures: bool,
    limit: int | None,
) -> pd.DataFrame:
    filtered = pred.copy()
    if method:
        filtered = filtered[filtered["method"] == method]
    if only_failures:
        if metrics is None:
            raise SystemExit("--only-failures requires --metrics")
        failed = metrics[metrics["end_to_end_success"] == False][["image_name", "method"]]  # noqa: E712
        filtered = filtered.merge(failed, on=["image_name", "method"], how="inner")
    filtered = filtered.sort_values(["method", "image_name"], kind="stable")
    if limit is not None:
        filtered = filtered.head(limit)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="annotations/plate_char_annotations.csv", help="Ground-truth CSV.")
    parser.add_argument("--pred", required=True, help="Prediction CSV.")
    parser.add_argument("--image-dir", default="dataset", help="Directory containing original images.")
    parser.add_argument("--out-dir", default="outputs/single_vis", help="Output directory.")
    parser.add_argument("--metrics", default="", help="Optional per_image_metrics.csv from evaluate.py.")
    parser.add_argument("--method", default="", help="Only visualize one method.")
    parser.add_argument("--only-failures", action="store_true", help="Only visualize failed rows from metrics.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of visualizations.")
    args = parser.parse_args()

    project_root = Path.cwd()
    gt, pred = load_inputs(Path(args.gt), Path(args.pred))
    gt_by_image = gt.set_index("image_name")
    metrics_path = Path(args.metrics) if args.metrics else None
    metrics = build_metrics_lookup(metrics_path)
    metrics_by_key = None
    if metrics is not None:
        metrics_by_key = metrics.set_index(["image_name", "method"])

    filtered = apply_filters(
        pred,
        metrics,
        args.method or None,
        args.only_failures,
        args.limit if args.limit and args.limit > 0 else None,
    )

    out_dir = Path(args.out_dir)
    image_dir = Path(args.image_dir)
    count = 0
    skipped = 0
    for _, pred_row in filtered.iterrows():
        image_name = pred_row["image_name"]
        method = str(pred_row["method"])
        if image_name not in gt_by_image.index:
            print(f"Skip unknown image_name: {image_name}")
            skipped += 1
            continue
        image = read_image(image_dir / image_name)
        if image is None:
            print(f"Skip missing/unreadable image: {image_dir / image_name}")
            skipped += 1
            continue
        metrics_row = None
        if metrics_by_key is not None and (image_name, method) in metrics_by_key.index:
            metric_value = metrics_by_key.loc[(image_name, method)]
            metrics_row = metric_value.iloc[0] if isinstance(metric_value, pd.DataFrame) else metric_value
        canvas = render_visualization(image, gt_by_image.loc[image_name], pred_row, metrics_row, project_root)
        output_path = out_dir / safe_filename(method) / f"{Path(image_name).stem}.png"
        write_image(output_path, canvas)
        count += 1

    print(f"Wrote {count} visualizations to {out_dir}")
    if skipped:
        print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
