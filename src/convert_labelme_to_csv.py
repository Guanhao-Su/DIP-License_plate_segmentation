"""Convert LabelMe JSON annotations to a CSV ground-truth table.

The dataset uses one `plate` rectangle and seven character rectangles
(`char_1` ... `char_7`) per image. Coordinates are exported as OpenCV-style
bounding boxes: [x, y, w, h].
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_CHAR_COUNT = 7
CHAR_LABEL_RE = re.compile(r"char_(\d+)$")


def bbox_from_points(points: list[list[float]]) -> list[int]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = round(min(xs))
    y1 = round(min(ys))
    x2 = round(max(xs))
    y2 = round(max(ys))
    return [x1, y1, x2 - x1, y2 - y1]


def infer_scene_type(image_name: str) -> str:
    if image_name.startswith("8月27日"):
        return "night_glare"
    if image_name.startswith("收费站"):
        return "toll_station"
    return "unknown"


def infer_split(image_name: str) -> str:
    """Create a fixed 10-image tuning split, then use the rest for testing."""
    tune_names = {
        "8月27日3时10分50秒0.bmp",
        "8月27日3时12分26秒0.bmp",
        "8月27日3时16分3秒0.bmp",
        "8月27日3时20分29秒0.bmp",
        "收费站0809.bmp",
        "收费站0813.bmp",
        "收费站0824.bmp",
        "收费站0854.bmp",
        "收费站0860.bmp",
        "收费站0863.bmp",
    }
    return "tune" if image_name in tune_names else "test"


def load_annotation(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = Path(data.get("imagePath") or json_path.with_suffix(".bmp").name).name
    shapes = data.get("shapes", [])

    plate_shapes = [shape for shape in shapes if shape.get("label") == "plate"]
    if len(plate_shapes) != 1:
        raise ValueError(f"{json_path.name}: expected 1 plate, got {len(plate_shapes)}")

    char_items: list[tuple[int, list[int]]] = []
    invalid_labels: list[str] = []
    for shape in shapes:
        label = str(shape.get("label", ""))
        if label == "plate":
            continue
        match = CHAR_LABEL_RE.fullmatch(label)
        if not match:
            invalid_labels.append(label)
            continue
        char_index = int(match.group(1))
        char_items.append((char_index, bbox_from_points(shape["points"])))

    if invalid_labels:
        raise ValueError(f"{json_path.name}: invalid labels: {sorted(set(invalid_labels))}")

    char_items.sort(key=lambda item: item[0])
    char_indices = [item[0] for item in char_items]
    expected_indices = list(range(1, EXPECTED_CHAR_COUNT + 1))
    if char_indices != expected_indices:
        raise ValueError(
            f"{json_path.name}: expected chars {expected_indices}, got {char_indices}"
        )

    char_bboxes = [bbox for _, bbox in char_items]
    row: dict[str, Any] = {
        "image_name": image_name,
        "image_width": int(data["imageWidth"]),
        "image_height": int(data["imageHeight"]),
        "split": infer_split(image_name),
        "scene_type": infer_scene_type(image_name),
        "plate_color": "",
        "plate_text": "",
        "plate_x": bbox_from_points(plate_shapes[0]["points"])[0],
        "plate_y": bbox_from_points(plate_shapes[0]["points"])[1],
        "plate_w": bbox_from_points(plate_shapes[0]["points"])[2],
        "plate_h": bbox_from_points(plate_shapes[0]["points"])[3],
        "char_count_gt": EXPECTED_CHAR_COUNT,
        "char_bboxes_gt": json.dumps(char_bboxes, ensure_ascii=False),
        "difficulty": "",
        "notes": "",
        "source_json": str(json_path.as_posix()),
    }

    for index, bbox in enumerate(char_bboxes, start=1):
        row[f"char_{index}_x"] = bbox[0]
        row[f"char_{index}_y"] = bbox[1]
        row[f"char_{index}_w"] = bbox[2]
        row[f"char_{index}_h"] = bbox[3]

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="dataset", help="Directory containing LabelMe JSON files.")
    parser.add_argument(
        "--output",
        default="annotations/plate_char_annotations.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    json_paths = sorted(input_dir.glob("*.json"), key=lambda path: path.name)
    if not json_paths:
        raise SystemExit(f"No JSON files found in {input_dir}")

    rows = [load_annotation(path) for path in json_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "image_name",
        "image_width",
        "image_height",
        "split",
        "scene_type",
        "plate_color",
        "plate_text",
        "plate_x",
        "plate_y",
        "plate_w",
        "plate_h",
        "char_count_gt",
        "char_bboxes_gt",
    ]
    char_fields = [
        f"char_{index}_{axis}"
        for index in range(1, EXPECTED_CHAR_COUNT + 1)
        for axis in ("x", "y", "w", "h")
    ]
    tail_fields = ["difficulty", "notes", "source_json"]
    fieldnames = base_fields + char_fields + tail_fields

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tune_count = sum(row["split"] == "tune" for row in rows)
    test_count = len(rows) - tune_count
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"split: tune={tune_count}, test={test_count}")


if __name__ == "__main__":
    main()
