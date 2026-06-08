"""Create small prediction CSV files used to smoke-test evaluate.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    gt_path = Path("annotations/plate_char_annotations.csv")
    out_dir = Path("results/eval_smoke_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")

    perfect_rows = []
    under_rows = []
    empty_rows = []
    for _, row in gt.iterrows():
        plate = [int(row["plate_x"]), int(row["plate_y"]), int(row["plate_w"]), int(row["plate_h"])]
        chars = json.loads(row["char_bboxes_gt"])
        perfect_rows.append(
            {
                "image_name": row["image_name"],
                "method": "perfect",
                "plate_bbox_pred": json.dumps(plate, ensure_ascii=False),
                "char_bboxes_pred": json.dumps(chars, ensure_ascii=False),
                "runtime_ms": 1.0,
                "status": "success",
            }
        )
        under_rows.append(
            {
                "image_name": row["image_name"],
                "method": "under_split",
                "plate_bbox_pred": json.dumps(plate, ensure_ascii=False),
                "char_bboxes_pred": json.dumps(chars[:-1], ensure_ascii=False),
                "runtime_ms": 1.0,
                "status": "success",
            }
        )
        empty_rows.append(
            {
                "image_name": row["image_name"],
                "method": "empty_plate",
                "plate_bbox_pred": "",
                "char_bboxes_pred": json.dumps(chars, ensure_ascii=False),
                "runtime_ms": 1.0,
                "status": "",
            }
        )

    pd.DataFrame(perfect_rows).to_csv(out_dir / "perfect.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(under_rows).to_csv(out_dir / "under_split.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(empty_rows).to_csv(out_dir / "empty_plate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(perfect_rows + under_rows + empty_rows).to_csv(
        out_dir / "combined.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Wrote smoke prediction inputs to {out_dir}")


if __name__ == "__main__":
    main()
