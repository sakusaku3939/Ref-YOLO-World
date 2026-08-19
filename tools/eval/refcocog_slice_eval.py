#!/usr/bin/env python3
"""Build RefCOCOg slice metrics from CustomRefCOCOMetric detailed results.

The script consumes the JSON written by CustomRefCOCOMetric(dump_results=True)
and the original RefCOCOg annotation file. It does not rerun inference; it only
summarizes already computed top-1 IoU/correctness into stable analysis slices.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SPATIAL_TERMS = {
    "left",
    "right",
    "leftmost",
    "rightmost",
    "top",
    "bottom",
    "topmost",
    "bottommost",
    "front",
    "behind",
    "between",
    "middle",
    "center",
    "centre",
    "next to",
    "near",
    "nearest",
    "above",
    "below",
    "under",
    "over",
}
LEFT_RIGHT_TERMS = {"left", "right", "leftmost", "rightmost"}
FRONT_BEHIND_TERMS = {"front", "behind"}
BETWEEN_TERMS = {"between", "middle", "center", "centre"}
TOP_BOTTOM_TERMS = {"top", "bottom", "topmost", "bottommost", "above", "below", "under", "over"}
ATTRIBUTE_TERMS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
    "small",
    "large",
    "big",
    "little",
    "tall",
    "short",
    "long",
    "striped",
    "spotted",
    "wooden",
    "metal",
    "standing",
    "sitting",
    "wearing",
    "holding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RefCOCOg Acc@0.5 by semantic slices")
    parser.add_argument("--ann-file", required=True, help="RefCOCOg annotation JSON")
    parser.add_argument(
        "--detailed-results",
        required=True,
        help="CustomRefCOCOMetric dump_results detailed_results.json",
    )
    parser.add_argument("--output-json", required=True, help="Output summary JSON path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--long-query-tokens", type=int, default=8)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def captions_for(item: dict[str, Any]) -> list[str]:
    captions = item.get("captions")
    if isinstance(captions, list) and captions:
        return [str(c) for c in captions]
    sentences = item.get("sentences")
    if isinstance(sentences, list):
        values = []
        for sent in sentences:
            if isinstance(sent, dict):
                values.append(str(sent.get("sent") or sent.get("raw") or ""))
        return [v for v in values if v]
    return []


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_set(captions: list[str]) -> set[str]:
    tokens: set[str] = set()
    for caption in captions:
        tokens.update(re.findall(r"[a-z0-9]+", normalize_text(caption)))
    return tokens


def contains_phrase(captions: list[str], terms: set[str]) -> bool:
    tokens = token_set(captions)
    if tokens & {term for term in terms if " " not in term}:
        return True
    normalized = [f" {normalize_text(caption)} " for caption in captions]
    return any(f" {term} " in caption for term in terms if " " in term for caption in normalized)


def max_query_len(captions: list[str]) -> int:
    return max((len(re.findall(r"[a-z0-9]+", caption.lower())) for caption in captions), default=0)


def image_key(item: dict[str, Any]) -> str:
    return str(item.get("global_image_id") or item.get("image_id") or item.get("file_name") or "")


def category_key(item: dict[str, Any]) -> str:
    return str(item.get("category_id", ""))


def build_context(annotations: list[dict[str, Any]]) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    image_counts: Counter[str] = Counter()
    category_counts: Counter[tuple[str, str]] = Counter()
    for item in annotations:
        img = image_key(item)
        cat = category_key(item)
        image_counts[img] += 1
        category_counts[(img, cat)] += 1
    return image_counts, category_counts


def metric_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "n_correct": 0,
            "acc_at_0_5": None,
            "mean_iou": None,
        }
    n_correct = sum(1 for row in records if row["is_correct"])
    return {
        "n": n,
        "n_correct": n_correct,
        "acc_at_0_5": round(n_correct / n * 100, 4),
        "mean_iou": round(mean(row["iou"] for row in records), 6),
    }


def coerce_result_index(result: dict[str, Any], fallback: int) -> int:
    raw = result.get("img_id", fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def build_rows(
    annotations: list[dict[str, Any]],
    detailed_results: list[dict[str, Any]],
    iou_threshold: float,
    long_query_tokens: int,
) -> list[dict[str, Any]]:
    image_counts, category_counts = build_context(annotations)
    rows: list[dict[str, Any]] = []
    for fallback_idx, result in enumerate(detailed_results):
        idx = coerce_result_index(result, fallback_idx)
        if idx < 0 or idx >= len(annotations):
            continue
        ann = annotations[idx]
        captions = captions_for(ann) or [str(c) for c in result.get("captions", [])]
        img = image_key(ann)
        cat = category_key(ann)
        iou = float(result.get("iou", 0.0) or 0.0)
        is_correct = bool(result.get("is_correct", iou >= iou_threshold))
        rows.append(
            {
                "idx": idx,
                "iou": iou,
                "is_correct": is_correct,
                "has_spatial_word": contains_phrase(captions, SPATIAL_TERMS),
                "has_left_right_word": contains_phrase(captions, LEFT_RIGHT_TERMS),
                "has_front_behind_word": contains_phrase(captions, FRONT_BEHIND_TERMS),
                "has_between_word": contains_phrase(captions, BETWEEN_TERMS),
                "has_top_bottom_word": contains_phrase(captions, TOP_BOTTOM_TERMS),
                "has_attribute_word": contains_phrase(captions, ATTRIBUTE_TERMS),
                "long_query": max_query_len(captions) >= long_query_tokens,
                "multi_ref_image": image_counts[img] > 1,
                "single_ref_image": image_counts[img] == 1,
                "same_category_image": category_counts[(img, cat)] > 1,
                "not_same_category_image": category_counts[(img, cat)] == 1,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slice_names = [
        "has_spatial_word",
        "no_spatial_word",
        "has_left_right_word",
        "no_left_right_word",
        "has_front_behind_word",
        "has_between_word",
        "has_top_bottom_word",
        "has_attribute_word",
        "no_attribute_word",
        "long_query",
        "short_query",
        "multi_ref_image",
        "single_ref_image",
        "same_category_image",
        "not_same_category_image",
    ]
    predicates = {
        "has_spatial_word": lambda r: r["has_spatial_word"],
        "no_spatial_word": lambda r: not r["has_spatial_word"],
        "has_left_right_word": lambda r: r["has_left_right_word"],
        "no_left_right_word": lambda r: not r["has_left_right_word"],
        "has_front_behind_word": lambda r: r["has_front_behind_word"],
        "has_between_word": lambda r: r["has_between_word"],
        "has_top_bottom_word": lambda r: r["has_top_bottom_word"],
        "has_attribute_word": lambda r: r["has_attribute_word"],
        "no_attribute_word": lambda r: not r["has_attribute_word"],
        "long_query": lambda r: r["long_query"],
        "short_query": lambda r: not r["long_query"],
        "multi_ref_image": lambda r: r["multi_ref_image"],
        "single_ref_image": lambda r: r["single_ref_image"],
        "same_category_image": lambda r: r["same_category_image"],
        "not_same_category_image": lambda r: r["not_same_category_image"],
    }
    return {
        "overall": metric_payload(rows),
        "slices": {
            name: metric_payload([row for row in rows if predicates[name](row)]) for name in slice_names
        },
    }


def main() -> int:
    args = parse_args()
    annotations = load_json(args.ann_file)
    detailed_results = load_json(args.detailed_results)
    if not isinstance(annotations, list):
        raise TypeError("annotation file must contain a list")
    if not isinstance(detailed_results, list):
        raise TypeError("detailed results file must contain a list")

    rows = build_rows(annotations, detailed_results, args.iou_threshold, args.long_query_tokens)
    summary = summarize(rows)
    summary["inputs"] = {
        "ann_file": args.ann_file,
        "detailed_results": args.detailed_results,
        "iou_threshold": args.iou_threshold,
        "long_query_tokens": args.long_query_tokens,
        "matched_results": len(rows),
        "annotation_count": len(annotations),
        "detailed_result_count": len(detailed_results),
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
