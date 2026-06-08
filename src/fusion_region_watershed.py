"""Member C algorithm runner: fused plate localization plus region/watershed chars.

This file keeps the shared CSV interface and CLI entrypoint, while the plate
detector and character segmenter live in dedicated modules.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fusion_region_watershed_chars import CharParams, save_debug_if_requested, run_char_segmentation
from fusion_region_watershed_common import (
    gt_plate_bbox,
    is_empty_value,
    load_ground_truth,
    load_plate_predictions,
    numeric_or_zero,
    parse_plate_bbox,
    read_image,
)
from fusion_region_watershed_plate import PlateParams, detect_plate


METHOD_NAME = "member_c_fusion_region_watershed_v1"


@dataclass(frozen=True)
class MemberCParams:
    plate: PlateParams = field(default_factory=PlateParams)
    chars: CharParams = field(default_factory=CharParams)


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
        default="results/fusion_region_watershed_v1_predictions.csv",
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
