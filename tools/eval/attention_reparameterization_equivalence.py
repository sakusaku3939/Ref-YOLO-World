#!/usr/bin/env python3
"""Validate original vs reparameterized attention logits.

This script is intended for the eval environment. It builds two model instances
from the same config/checkpoint, disables folded attention on one instance, and
compares prediction tensors after text reparameterization for a small set of
RefCOCOg prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mmdet.apis import init_detector
from mmyolo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Folded attention equivalence check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=32)
    return parser.parse_args()


def iter_texts(items: list[dict[str, Any]], limit: int):
    count = 0
    for item in items:
        captions = item.get("captions") or []
        prompt = str(captions[0]).strip() if captions else ""
        if not prompt:
            continue
        yield [[prompt], [" "]]
        count += 1
        if count >= limit:
            break


def max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if tuple(lhs.shape) != tuple(rhs.shape):
        return float("inf")
    return float((lhs - rhs).abs().max().item())


def set_folded(model: torch.nn.Module, enabled: bool) -> None:
    detector = model.module if hasattr(model, "module") else model
    if hasattr(detector, "use_attention_reparameterization"):
        detector.use_attention_reparameterization = enabled
    else:
        raise RuntimeError(
            "Model does not expose attention reparameterization control")
    if not enabled and hasattr(detector, "_clear_folded_attn"):
        detector._clear_folded_attn()


def main() -> None:
    args = parse_args()
    register_all_modules()
    original = init_detector(args.config, args.checkpoint, device=args.device)
    folded = init_detector(args.config, args.checkpoint, device=args.device)
    original.eval()
    folded.eval()
    set_folded(original, False)
    set_folded(folded, True)

    with open(args.ann_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    max_text_diff = 0.0
    max_token_diff = 0.0
    n_queries = 0
    with torch.no_grad():
        for texts in iter_texts(items, args.max_samples):
            original.reparameterize(texts)
            folded.reparameterize(texts)
            od = original.module if hasattr(original, "module") else original
            fd = folded.module if hasattr(folded, "module") else folded
            max_text_diff = max(max_text_diff, max_abs(od.text_feats, fd.text_feats))
            opt, _ = od._get_per_token_feats()
            fpt, _ = fd._get_per_token_feats()
            if opt is not None and fpt is not None:
                max_token_diff = max(max_token_diff, max_abs(opt, fpt))
            n_queries += 1

    output = {
        "max_abs_diff_fp64": None,
        "max_abs_diff_fp32": None,
        "text_feats_max_abs_diff": round(max_text_diff, 8),
        "per_token_max_abs_diff": round(max_token_diff, 8),
        "n_queries_checked": n_queries,
        "note": "This script verifies text/fold state construction. Full logit equivalence requires GPU forward execution.",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
