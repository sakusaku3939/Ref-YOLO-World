#!/usr/bin/env python3
"""Aggregate LoRA-merge evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LoRA merge eval artifacts")
    parser.add_argument("--speed", default="results/lora_merge/speed.json")
    parser.add_argument("--merge-equivalence", default="results/lora_merge/merge_equivalence.json")
    parser.add_argument("--reparam-equivalence", default="results/lora_merge/reparam_equivalence.json")
    parser.add_argument(
        "--detailed",
        default="results/lora_merge/formal_eval/detailed_results.json",
    )
    parser.add_argument(
        "--baseline-detailed",
        default="results/refcocog_detailed_results/detailed_results.json",
    )
    parser.add_argument("--flip-audit", default="results/lora_merge/flip_audit.json")
    parser.add_argument("--onnx-export-check", default="results/lora_merge/onnx_export_check.json")
    parser.add_argument(
        "--formal-summary-output",
        default="results/lora_merge/formal_eval_summary.json",
    )
    parser.add_argument("--output", default="results/lora_merge/summary.json")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_detailed(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("is_correct"))
    acc_pct = round(correct / total * 100.0, 4) if total else 0.0
    return {
        "n_total": total,
        "n_correct": correct,
        "acc_at_0_5_pct": acc_pct,
    }


def main() -> None:
    args = parse_args()
    speed = load_json(args.speed)
    merge_equivalence = load_json(args.merge_equivalence)
    reparam_equivalence = load_json(args.reparam_equivalence)
    detailed = load_json(args.detailed)
    baseline_detailed = load_json(args.baseline_detailed)
    flip_audit = load_json(args.flip_audit)
    onnx_export = load_json(args.onnx_export_check)

    e2e_ms = float(speed["e2e_sync_ms"]["mean"])
    reparam_ms = float(speed["reparam_sync_ms"]["mean"])
    yolo_ms = float(speed["yolo_sync_ms"]["mean"])
    detailed_summary = summarize_detailed(detailed)
    baseline_summary = summarize_detailed(baseline_detailed)

    formal_summary_path = Path(args.formal_summary_output)
    formal_summary_path.parent.mkdir(parents=True, exist_ok=True)
    formal_summary_path.write_text(
        json.dumps(detailed_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failures: list[str] = []
    if merge_equivalence["merge_text_max_abs_diff"] >= 1e-5:
        failures.append(
            f"merge_text_max_abs_diff={merge_equivalence['merge_text_max_abs_diff']} >= 1e-5")
    if reparam_equivalence["reparam_state_max_abs_diff"] >= 1e-5:
        failures.append(
            f"reparam_state_max_abs_diff={reparam_equivalence['reparam_state_max_abs_diff']} >= 1e-5")
    if flip_audit["n_common"] != 2573:
        failures.append(f"n_common={flip_audit['n_common']} != 2573")
    if flip_audit["correct_delta"] < -2:
        failures.append(f"correct_delta={flip_audit['correct_delta']} < -2")
    if flip_audit["flips_down"] > 2:
        failures.append(f"flips_down={flip_audit['flips_down']} > 2")
    if not bool(onnx_export.get("onnx_exportable")):
        failures.append("onnx_exportable=false")

    output = {
        "speed": speed,
        "merge_equivalence": merge_equivalence,
        "reparam_equivalence": reparam_equivalence,
        "formal_eval": {
            "detailed_results_path": args.detailed,
            "summary": detailed_summary,
        },
        "baseline_eval": {
            "detailed_results_path": args.baseline_detailed,
            "summary": baseline_summary,
        },
        "flip_audit": flip_audit,
        "onnx_export_check": onnx_export,
        "e2e_sync_ms_mean": round(e2e_ms, 4),
        "reparam_ms_mean": round(reparam_ms, 4),
        "yolo_ms_mean": round(yolo_ms, 4),
        "acc_at_0_5_pct": detailed_summary["acc_at_0_5_pct"],
        "n_correct": detailed_summary["n_correct"],
        "n_total": detailed_summary["n_total"],
        "merge_text_max_abs_diff": merge_equivalence["merge_text_max_abs_diff"],
        "reparam_state_max_abs_diff": reparam_equivalence["reparam_state_max_abs_diff"],
        "correct_delta": flip_audit["correct_delta"],
        "flips_down": flip_audit["flips_down"],
        "flips_up": flip_audit["flips_up"],
        "n_common": flip_audit["n_common"],
        "onnx_exportable": bool(onnx_export.get("onnx_exportable")),
        "thresholds": {
            "expected_samples": 2573,
            "merge_text_max_abs_diff": 1e-5,
            "reparam_state_max_abs_diff": 1e-5,
            "min_correct_delta": -2,
            "max_flips_down": 2,
        },
        "failures": failures,
        "overall_pass": not failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "e2e_sync_ms_mean": output["e2e_sync_ms_mean"],
        "acc_at_0_5_pct": output["acc_at_0_5_pct"],
        "merge_text_max_abs_diff": output["merge_text_max_abs_diff"],
        "reparam_state_max_abs_diff": output["reparam_state_max_abs_diff"],
        "correct_delta": output["correct_delta"],
        "flips_down": output["flips_down"],
        "onnx_exportable": output["onnx_exportable"],
        "overall_pass": output["overall_pass"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
