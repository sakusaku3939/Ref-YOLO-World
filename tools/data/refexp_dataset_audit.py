#!/usr/bin/env python3
"""Audit RefCOCO-family annotations and train/evaluation image separation.

Refactored from:
https://github.com/sakusaku3939/YOLO-World-test/blob/master/tools/research/refexp_dataset_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_utils import load_json, write_json
from filter_refexp_train_overlap import EVAL_FILES, TRAIN_FILES


def captions_for(record: dict[str, Any]) -> list[str]:
    captions = record.get("captions")
    if isinstance(captions, list):
        values = [str(value).strip() for value in captions]
        if any(values):
            return [value for value in values if value]
    sentences = record.get("sentences")
    if not isinstance(sentences, list):
        return []
    values = []
    for sentence in sentences:
        if isinstance(sentence, dict):
            text = sentence.get("sent") or sentence.get("raw") or ""
        else:
            text = sentence
        if str(text).strip():
            values.append(str(text).strip())
    return values


def _numbers(values: object) -> list[float] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    result = [float(value) for value in values]
    return result if all(math.isfinite(value) for value in result) else None


def valid_bbox(record: dict[str, Any]) -> bool:
    bbox = _numbers(record.get("bbox"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return x2 > x1 and y2 > y1

    raw_annotation = record.get("raw_anns", {})
    if isinstance(raw_annotation, str):
        try:
            raw_annotation = json.loads(raw_annotation)
        except json.JSONDecodeError:
            return False
    if not isinstance(raw_annotation, dict):
        return False
    bbox = _numbers(raw_annotation.get("bbox"))
    if bbox is None:
        return False
    _, _, width, height = bbox
    return width > 0 and height > 0


def resolve_image(record: dict[str, Any], data_root: Path) -> Path:
    explicit = record.get("image_path") or record.get("img_path")
    if explicit:
        candidate = Path(str(explicit))
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "data":
            return data_root.parent / candidate
        return data_root / candidate
    image_id = int(record["image_id"])
    return data_root / "coco" / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"


def audit_file(path: Path, data_root: Path, check_images: bool) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "status": "missing",
            "records": 0,
            "unique_images": 0,
            "invalid_bbox": 0,
            "records_without_text": 0,
            "missing_images": 0,
            "image_ids": [],
        }
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list")

    image_ids = set()
    invalid_bbox = 0
    records_without_text = 0
    missing_images = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "image_id" not in record:
            raise ValueError(f"{path}: invalid record at index {index}")
        image_ids.add(str(record["image_id"]))
        invalid_bbox += int(not valid_bbox(record))
        records_without_text += int(not captions_for(record))
        if check_images:
            missing_images += int(not resolve_image(record, data_root).is_file())

    return {
        "path": str(path),
        "status": "ok",
        "records": len(records),
        "unique_images": len(image_ids),
        "invalid_bbox": invalid_bbox,
        "records_without_text": records_without_text,
        "missing_images": missing_images,
        "image_ids": sorted(image_ids),
    }


def run_audit(data_root: Path, *, check_images: bool = False) -> dict[str, Any]:
    datasets = {}
    for name, relative_path in {**TRAIN_FILES, **EVAL_FILES}.items():
        datasets[name] = audit_file(
            data_root / relative_path,
            data_root,
            check_images,
        )

    evaluation_union = set()
    for name in EVAL_FILES:
        evaluation_union.update(datasets[name]["image_ids"])

    overlaps = {}
    for train_name in TRAIN_FILES:
        train_images = set(datasets[train_name]["image_ids"])
        overlaps[train_name] = {
            eval_name: len(train_images & set(datasets[eval_name]["image_ids"]))
            for eval_name in EVAL_FILES
        }
        overlaps[train_name]["evaluation_union"] = len(
            train_images & evaluation_union
        )

    missing_annotations = [
        summary["path"]
        for summary in datasets.values()
        if summary["status"] == "missing"
    ]
    invalid_bbox = sum(summary["invalid_bbox"] for summary in datasets.values())
    records_without_text = sum(
        summary["records_without_text"] for summary in datasets.values()
    )
    missing_images = sum(summary["missing_images"] for summary in datasets.values())
    remaining_overlap_images = sum(
        summary["evaluation_union"] for summary in overlaps.values()
    )
    hypothesis_pass = not any(
        (
            missing_annotations,
            invalid_bbox,
            records_without_text,
            remaining_overlap_images,
            missing_images if check_images else 0,
        )
    )
    return {
        "hypothesis_pass": hypothesis_pass,
        "policy": "all train images must be absent from all eight evaluation splits",
        "datasets": datasets,
        "overlap_images": overlaps,
        "summary": {
            "missing_annotations": missing_annotations,
            "invalid_bbox": invalid_bbox,
            "records_without_text": records_without_text,
            "missing_images": missing_images,
            "remaining_overlap_images": remaining_overlap_images,
            "evaluation_unique_images_union": len(evaluation_union),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all RefCOCO-family splits and image-level separation."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/refexp_audit.json"),
    )
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_audit(args.data_root, check_images=args.check_images)
        write_json(args.output, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["hypothesis_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
