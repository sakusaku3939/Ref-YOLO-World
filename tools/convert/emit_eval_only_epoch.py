#!/usr/bin/env python3
"""Emit a mmengine-style eval epoch line for eval-only experiment monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument(
        "--formal-summary",
        default=None,
        help="Optional path to formal_eval_summary.json for n_correct/n_total fallback",
    )
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    summary = load_json(args.summary)
    formal_summary = load_json(args.formal_summary) if args.formal_summary else {}

    acc = summary.get("acc_at_0_5_pct", formal_summary.get("acc_at_0_5_pct"))
    n_correct = summary.get("n_correct", formal_summary.get("n_correct"))
    n_total = summary.get("n_total", formal_summary.get("n_total"))

    if acc is None:
        raise SystemExit("acc_at_0_5_pct not found in summary inputs")

    parts = [
        f"Epoch(val) [{args.epoch}][1/1]",
        f"refcoco/Acc@0.5: {float(acc):.4f}",
    ]
    if n_correct is not None:
        parts.append(f"refcoco/n_correct: {int(n_correct)}")
    if n_total is not None:
        parts.append(f"refcoco/n_total: {int(n_total)}")
    print("  ".join(parts))


if __name__ == "__main__":
    main()
