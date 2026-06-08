"""License-plate localization with Sobel edges and morphology.

This script implements the first baseline detector:

1. grayscale + CLAHE
2. Gaussian blur
3. Sobel-x gradient + Otsu threshold
4. morphology open/close/dilate
5. contour filtering and candidate scoring

The output CSV follows the shared prediction interface used by evaluate.py and
visualize_single.py. Character segmentation is not connected yet, so
char_bboxes_pred is always [].
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


METHOD_NAME = "edge_morph_plate_v1"


@dataclass(frozen=True)
class DetectorParams:
    clahe_clip: float = 2.0
    clahe_tile: tuple[int, int] = (8, 8)
    blur_kernel: int = 7
    sobel_kernel: int = 3
    open_kernel: tuple[int, int] = (3, 3)
    close_kernel: tuple[int, int] = (17, 3)
    dilate_kernel: tuple[int, int] = (3, 3)
    min_area: int = 1000
    min_width: int = 70
    max_width: int = 280
    min_height: int = 25
    max_height: int = 120
    aspect_min: float = 2.0
    aspect_max: float = 4.8
    target_aspect: float = 2.85
    target_width: int = 162
    target_height: int = 57
    score_weight_aspect: float = 0.30
    score_weight_density: float = 0.25
    score_weight_size: float = 0.20
    score_weight_rectangularity: float = 0.15
    score_weight_position: float = 0.10


@dataclass
class Candidate:
    score: float
    bbox: list[int]
    aspect: float
    edge_density: float
    rectangularity: float


def read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def kernel(size: tuple[int, int]) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)


def preprocess(gray_bgr: np.ndarray, params: DetectorParams) -> np.ndarray:
    gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=params.clahe_tile)
    gray = clahe.apply(gray)
    if params.blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (params.blur_kernel, params.blur_kernel), 0)
    return gray


def sobel_otsu_edges(gray: np.ndarray, params: DetectorParams) -> tuple[np.ndarray, np.ndarray]:
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=params.sobel_kernel)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return grad_x, binary


def morphology(edge_binary: np.ndarray, params: DetectorParams) -> np.ndarray:
    cleaned = cv2.morphologyEx(edge_binary, cv2.MORPH_OPEN, kernel(params.open_kernel), iterations=1)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel(params.close_kernel), iterations=1)
    closed = cv2.dilate(closed, kernel(params.dilate_kernel), iterations=1)
    return closed


def score_candidate(
    bbox: list[int],
    contour_area: float,
    edge_binary: np.ndarray,
    image_shape: tuple[int, int],
    params: DetectorParams,
) -> Candidate | None:
    x, y, w, h = bbox
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

    rectangularity = contour_area / area if area else 0.0
    roi_edges = edge_binary[y : y + h, x : x + w]
    edge_density = float(np.count_nonzero(roi_edges)) / area if area else 0.0

    height = image_shape[0]
    center_y_ratio = (y + h / 2.0) / height
    aspect_score = max(0.0, 1.0 - abs(aspect - params.target_aspect) / 2.2)
    size_score = max(0.0, 1.0 - abs(w - params.target_width) / 140.0) * max(
        0.0, 1.0 - abs(h - params.target_height) / 80.0
    )
    rectangularity_score = min(1.0, rectangularity / 0.55)
    density_score = min(1.0, edge_density / 0.18)
    position_score = 1.0 - min(1.0, abs(center_y_ratio - 0.52) / 0.55)

    score = (
        params.score_weight_aspect * aspect_score
        + params.score_weight_density * density_score
        + params.score_weight_size * size_score
        + params.score_weight_rectangularity * rectangularity_score
        + params.score_weight_position * position_score
    )
    return Candidate(
        score=score,
        bbox=bbox,
        aspect=aspect,
        edge_density=edge_density,
        rectangularity=rectangularity,
    )


def detect_plate(image: np.ndarray, params: DetectorParams) -> tuple[Candidate | None, list[Candidate]]:
    gray = preprocess(image, params)
    edge_strength, edge_binary = sobel_otsu_edges(gray, params)
    morph = morphology(edge_binary, params)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Candidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        candidate = score_candidate(
            [int(x), int(y), int(w), int(h)],
            float(cv2.contourArea(contour)),
            edge_binary=edge_binary,
            image_shape=edge_strength.shape,
            params=params,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0] if candidates else None
    return best, candidates


def params_json(params: DetectorParams, method_name: str, preset: str) -> str:
    values: dict[str, Any] = asdict(params)
    values["method_name"] = method_name
    values["preset"] = preset
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def params_for_preset(preset: str) -> DetectorParams:
    if preset == "base":
        return DetectorParams()
    if preset == "rect_heavy":
        return DetectorParams(
            score_weight_aspect=0.22,
            score_weight_density=0.18,
            score_weight_size=0.12,
            score_weight_rectangularity=0.40,
            score_weight_position=0.08,
        )
    raise ValueError(f"Unknown preset: {preset}")


def load_ground_truth(path: Path, split: str) -> pd.DataFrame:
    gt = pd.read_csv(path, encoding="utf-8-sig")
    required = {"image_name", "split"}
    missing = required - set(gt.columns)
    if missing:
        raise SystemExit(f"Ground-truth CSV is missing columns: {sorted(missing)}")
    if split != "all":
        gt = gt[gt["split"] == split].copy()
    if gt.empty:
        raise SystemExit(f"No rows found for split={split}")
    return gt


def make_prediction_row(
    image_name: str,
    image_dir: Path,
    params: DetectorParams,
    params_text: str,
    method_name: str,
) -> dict[str, Any]:
    start = time.perf_counter()
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
        }

    best, candidates = detect_plate(image, params)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if best is None:
        return {
            "image_name": image_name,
            "method": method_name,
            "plate_bbox_pred": "",
            "char_bboxes_pred": "[]",
            "params": params_text,
            "runtime_ms": runtime_ms,
            "status": "plate_not_found",
            "failure_reason": "no_candidate_after_filtering",
        }

    return {
        "image_name": image_name,
        "method": method_name,
        "plate_bbox_pred": json.dumps(best.bbox, ensure_ascii=False),
        "char_bboxes_pred": "[]",
        "params": params_text,
        "runtime_ms": runtime_ms,
        "status": "success",
        "failure_reason": "",
        "candidate_score": best.score,
        "candidate_count": len(candidates),
        "candidate_aspect": best.aspect,
        "candidate_edge_density": best.edge_density,
        "candidate_rectangularity": best.rectangularity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="annotations/plate_char_annotations.csv", help="Ground-truth CSV.")
    parser.add_argument("--image-dir", default="dataset", help="Directory containing BMP images.")
    parser.add_argument("--out", default="results/edge_morph_plate_v1_predictions.csv", help="Output CSV.")
    parser.add_argument("--split", choices=["all", "tune", "test"], default="all", help="Subset to run.")
    parser.add_argument("--method-name", default=METHOD_NAME, help="Method name written to the prediction CSV.")
    parser.add_argument(
        "--preset",
        choices=["base", "rect_heavy"],
        default="base",
        help="Parameter preset. base preserves the original v1 behavior.",
    )
    args = parser.parse_args()

    params = params_for_preset(args.preset)
    params_text = params_json(params, args.method_name, args.preset)
    gt = load_ground_truth(Path(args.gt), args.split)
    image_dir = Path(args.image_dir)

    rows = [
        make_prediction_row(str(row["image_name"]), image_dir, params, params_text, args.method_name)
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
