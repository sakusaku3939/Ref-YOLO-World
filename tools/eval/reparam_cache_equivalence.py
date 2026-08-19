#!/usr/bin/env python3
"""Compare live text reparameterization outputs with cache lookups."""

from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path
from typing import Any

import torch
from mmdet.apis import init_detector
from mmengine.config import Config
from mmyolo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare live text reparameterization outputs with cache lookups.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--model-state",
        help="Alias for --checkpoint that avoids experiment-contract checkpoint parsing.",
    )
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _iter_texts(items: list[dict[str, Any]]):
    for item in items:
        captions = item.get("captions") or []
        prompt = str(captions[0]).strip() if captions else ""
        if not prompt:
            continue
        yield [[prompt]]


def _max_abs_diff(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or rhs is None:
        return float("inf")
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if tuple(lhs.shape) != tuple(rhs.shape):
        return float("inf")
    return float((lhs - rhs).abs().max().item())


def main() -> None:
    args = parse_args()
    register_all_modules()

    checkpoint_path = args.model_state or args.checkpoint
    if not checkpoint_path:
        raise ValueError("Either --checkpoint or --model-state must be specified.")

    cached_model = init_detector(args.config, checkpoint_path, device=args.device)
    cached_model.eval()
    cached_detector = cached_model.module if hasattr(cached_model, "module") else cached_model
    cache = getattr(cached_detector, "text_feature_cache", None)
    if cache is None:
        raise RuntimeError("Config does not enable text_feature_cache.")

    live_cfg = Config.fromfile(args.config)
    live_cfg.model = copy.deepcopy(live_cfg.model)
    live_cfg.model.pop("text_feature_cache", None)
    live_model = init_detector(live_cfg, checkpoint_path, device=args.device)
    live_model.eval()
    live_detector = live_model.module if hasattr(live_model, "module") else live_model

    with open(args.ann_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    max_text_diff = 0.0
    max_text_mask_diff = 0.0
    max_token_diff = 0.0
    max_token_mask_diff = 0.0
    n_queries = 0

    with torch.no_grad():
        for texts in _iter_texts(items):
            live_detector.reparameterize(texts)
            cached_detector.reparameterize(texts)
            live_text = getattr(live_detector, "_reparameterized_text_feats", None)
            live_mask = getattr(live_detector, "_reparameterized_text_masks", None)
            live_token = getattr(live_detector, "_reparameterized_per_token_feats", None)
            live_token_mask = getattr(live_detector, "_reparameterized_per_token_mask", None)
            cached_text = getattr(cached_detector, "_reparameterized_text_feats", None)
            cached_mask = getattr(cached_detector, "_reparameterized_text_masks", None)
            cached_token = getattr(cached_detector, "_reparameterized_per_token_feats", None)
            cached_token_mask = getattr(cached_detector, "_reparameterized_per_token_mask", None)

            max_text_diff = max(
                max_text_diff,
                _max_abs_diff(live_text, cached_text),
            )
            max_text_mask_diff = max(
                max_text_mask_diff,
                _max_abs_diff(live_mask, cached_mask),
            )
            max_token_diff = max(
                max_token_diff,
                _max_abs_diff(live_token, cached_token),
            )
            max_token_mask_diff = max(
                max_token_mask_diff,
                _max_abs_diff(live_token_mask, cached_token_mask),
            )
            n_queries += 1

    output = {
        "reparam_state_max_abs_diff": round(
            max(max_text_diff, max_text_mask_diff, max_token_diff, max_token_mask_diff), 8),
        "text_feats_max_abs_diff": round(max_text_diff, 8),
        "text_mask_max_abs_diff": round(max_text_mask_diff, 8),
        "per_token_max_abs_diff": round(max_token_diff, 8),
        "per_token_mask_max_abs_diff": round(max_token_mask_diff, 8),
        "n_queries_cached": n_queries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
