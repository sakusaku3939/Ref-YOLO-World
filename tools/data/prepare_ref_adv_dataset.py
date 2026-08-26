#!/usr/bin/env python3
"""Download Ref-Adv-s and convert it to Ref-YOLO-World JSON records.

The dataset is distributed by dddraxxx/ref-adv-s under CC-BY-4.0.
Conversion logic is adapted from:
https://github.com/sakusaku3939/YOLO-World-test/blob/master/tools/research/prepare_generalized_rec_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_utils import download_file, fetch_hf_rows, write_json


DATASET_ID = "dddraxxx/ref-adv-s"
DATASET_URL = "https://huggingface.co/datasets/dddraxxx/ref-adv-s"
DATASET_LICENSE = "CC-BY-4.0"
EXPECTED_ROWS = 1142
FetchRows = Callable[..., list[dict]]
Download = Callable[..., None]


def _xyxy(row: dict, index: int) -> list[float]:
    bbox = row.get("solution")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Ref-Adv-s row {index}: solution must have four values")
    if any(not isinstance(value, (int, float)) for value in bbox):
        raise ValueError(f"Ref-Adv-s row {index}: solution must be numeric")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = int(row["width"])
    height = int(row["height"])
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Ref-Adv-s row {index}: solution is outside the image")
    return [x1, y1, x2, y2]


def prepare_ref_adv(
    data_root: Path,
    *,
    force: bool = False,
    expected_rows: int = EXPECTED_ROWS,
    fetch_rows: FetchRows = fetch_hf_rows,
    download: Download = download_file,
    fetch_options: dict | None = None,
) -> tuple[list[dict], dict]:
    fetch_options = fetch_options or {}
    rows = fetch_rows(DATASET_ID, "train", **fetch_options)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Ref-Adv-s: expected {expected_rows} rows, got {len(rows)}"
        )

    output_dir = data_root / "ref_adv"
    image_dir = output_dir / "images"
    records = []
    for index, row in enumerate(rows):
        image = row.get("image") or {}
        image_url = image.get("src") if isinstance(image, dict) else None
        if not image_url:
            raise ValueError(f"Ref-Adv-s row {index}: image URL is missing")
        image_name = Path(str(row.get("image_file") or f"{index:06d}.jpg")).name
        image_path = image_dir / image_name
        download(str(image_url), image_path, force=force)

        caption = str(row.get("normal_caption") or "").strip()
        if not caption:
            raise ValueError(f"Ref-Adv-s row {index}: caption is missing")
        x1, y1, x2, y2 = _xyxy(row, index)
        image_id = int(row.get("row_idx", index))
        records.append(
            {
                "image_id": image_id,
                "image_path": str(Path("ref_adv/images") / image_name),
                "file_name": image_name,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "sentences": [{"sent": caption, "raw": caption}],
                "captions": [caption],
                "ann_id": image_id,
                "raw_anns": {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "image_id": image_id,
                    "id": image_id,
                    "category_id": 0,
                    "iscrowd": 0,
                },
                "bbox": [x1, y1, x2, y2],
                "bbox_format": "xyxy",
                "image_source": row.get("image_source"),
                "use_negation": bool(row.get("use_negation")),
                "distractor_count": int(row.get("distractors") or 0),
                "human_authored": bool(row.get("human_authored")),
            }
        )

    manifest = {
        "source": {
            "dataset": DATASET_ID,
            "url": DATASET_URL,
            "license": DATASET_LICENSE,
        },
        "records": len(records),
        "images": len({record["image_path"] for record in records}),
        "output": str(output_dir / "ref_adv_s.json"),
    }
    write_json(output_dir / "ref_adv_s.json", records)
    write_json(output_dir / "source_manifest.json", manifest)
    return records, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Ref-Adv-s evaluation set.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--sleep-sec", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--backoff-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _, manifest = prepare_ref_adv(
            args.data_root,
            force=args.force,
            fetch_options={
                "chunk_size": args.chunk_size,
                "sleep_sec": args.sleep_sec,
                "max_retries": args.max_retries,
                "backoff_sec": args.backoff_sec,
            },
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
