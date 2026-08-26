#!/usr/bin/env python3
"""Prepare leak-free RefCOCO, RefCOCO+ and RefCOCOg annotations.

The Hugging Face rows API is used so the script has no dataset-library
dependency. All published evaluation splits are materialized before their
image-ID union is removed from the three training files.

Refactored from:
https://github.com/sakusaku3939/YOLO-World-test/blob/master/tools/research/prepare_refcoco_multi_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_utils import fetch_hf_rows, load_json, write_json
from filter_refexp_train_overlap import build_outputs


@dataclass(frozen=True)
class Target:
    name: str
    dataset: str
    split: str
    output: Path
    expected_rows: int
    training: bool = False


TARGETS = (
    Target(
        "refcoco_train",
        "jxu124/refcoco",
        "train",
        Path("refcoco/refcoco/train.json"),
        42404,
        True,
    ),
    Target(
        "refcoco_val",
        "jxu124/refcoco",
        "validation",
        Path("refcoco/refcoco/validation.json"),
        3811,
    ),
    Target(
        "refcoco_testA",
        "jxu124/refcoco",
        "test",
        Path("refcoco/refcoco/testA.json"),
        1975,
    ),
    Target(
        "refcoco_testB",
        "jxu124/refcoco",
        "testB",
        Path("refcoco/refcoco/testB.json"),
        1810,
    ),
    Target(
        "refcoco+_train",
        "jxu124/refcocoplus",
        "train",
        Path("refcoco/refcoco+/train.json"),
        42278,
        True,
    ),
    Target(
        "refcoco+_val",
        "jxu124/refcocoplus",
        "validation",
        Path("refcoco/refcoco+/validation.json"),
        3805,
    ),
    Target(
        "refcoco+_testA",
        "jxu124/refcocoplus",
        "test",
        Path("refcoco/refcoco+/testA.json"),
        1975,
    ),
    Target(
        "refcoco+_testB",
        "jxu124/refcocoplus",
        "testB",
        Path("refcoco/refcoco+/testB.json"),
        1798,
    ),
    Target(
        "refcocog_train",
        "jxu124/refcocog",
        "train",
        Path("refcoco/refcocog/train.json"),
        42226,
        True,
    ),
    Target(
        "refcocog_val",
        "jxu124/refcocog",
        "validation",
        Path("refcoco/refcocog/validation.json"),
        2573,
    ),
    Target(
        "refcocog_test",
        "jxu124/refcocog",
        "test",
        Path("refcoco/refcocog/test.json"),
        5023,
    ),
)

REQUIRED_KEYS = ("image_id", "sentences", "raw_anns", "ann_id")
FetchRows = Callable[..., list[dict]]


def validate_rows(target: Target, rows: object, *, existing: bool) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{target.output}: expected a JSON list")
    row_count = len(rows)
    valid_existing_train = (
        existing and target.training and 0 < row_count <= target.expected_rows
    )
    if row_count != target.expected_rows and not valid_existing_train:
        raise ValueError(
            f"{target.output}: expected {target.expected_rows} rows, got {row_count}"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{target.output}: row {index} is not an object")
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"{target.output}: row {index} missing {missing}")
        if not isinstance(row["sentences"], list) or not row["sentences"]:
            raise ValueError(f"{target.output}: row {index} has no sentences")
    return rows


def materialize_target(
    target: Target,
    data_root: Path,
    *,
    force: bool,
    fetch_rows: FetchRows,
    fetch_options: dict,
) -> tuple[list[dict], str]:
    destination = data_root / target.output
    if destination.is_file() and not force:
        rows = validate_rows(target, load_json(destination), existing=True)
        return rows, "reused"

    rows = fetch_rows(
        target.dataset,
        target.split,
        **fetch_options,
    )
    rows = validate_rows(target, rows, existing=False)
    write_json(destination, rows)
    return rows, "downloaded"


def prepare_refcoco_family(
    data_root: Path,
    manifest_path: Path,
    *,
    force: bool = False,
    targets: tuple[Target, ...] = TARGETS,
    fetch_rows: FetchRows = fetch_hf_rows,
    fetch_options: dict | None = None,
) -> dict:
    fetch_options = fetch_options or {}
    prepared = {}
    for target in targets:
        rows, action = materialize_target(
            target,
            data_root,
            force=force,
            fetch_rows=fetch_rows,
            fetch_options=fetch_options,
        )
        prepared[target.name] = {
            "dataset": target.dataset,
            "source_split": target.split,
            "output": str(data_root / target.output),
            "rows_before_filter": len(rows),
            "action": action,
        }

    outputs, manifest = build_outputs(data_root, data_root / "refcoco")
    for output, rows in outputs.items():
        write_json(output, rows)

    manifest["hypothesis_pass"] = all(
        summary["remaining_overlap_images"] == 0
        for summary in manifest["training"].values()
    )
    manifest["preparation"] = prepared
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all RefCOCO-family splits and remove every evaluation "
            "image from the training annotations."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/refcoco_overlap_filter/overlap_manifest.json"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--sleep-sec", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--backoff-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_refcoco_family(
            args.data_root,
            args.manifest,
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
        RuntimeError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
