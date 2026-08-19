#!/usr/bin/env python3
"""Evaluate one checkpoint on all formal RefCOCO-family splits."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmengine.config import Config, ConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class SplitSpec:
    key: str
    column: str
    ann_file: str


SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec("refcoco_val", "RefCOCO val", "data/refcoco/refcoco/validation.json"),
    SplitSpec("refcoco_testA", "RefCOCO testA", "data/refcoco/refcoco/testA.json"),
    SplitSpec("refcoco_testB", "RefCOCO testB", "data/refcoco/refcoco/testB.json"),
    SplitSpec("refcoco_plus_val", "RefCOCO+ val", "data/refcoco/refcoco+/validation.json"),
    SplitSpec("refcoco_plus_testA", "RefCOCO+ testA", "data/refcoco/refcoco+/testA.json"),
    SplitSpec("refcoco_plus_testB", "RefCOCO+ testB", "data/refcoco/refcoco+/testB.json"),
    SplitSpec("refcocog_val", "RefCOCOg val", "data/refcoco/refcocog/validation.json"),
    SplitSpec("refcocog_test", "RefCOCOg test", "data/refcoco/refcocog/test.json"),
)
ACC_PATTERN = re.compile(
    r"(?:refcoco/)?Acc(?:uracy)?@0\.5['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
COUNT_PATTERN = re.compile(r"Correct:\s*[0-9]+\s*/\s*([0-9]+)")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel_ann_file(ann_file: str) -> str:
    if not ann_file.startswith("data/"):
        raise ValueError(f"Expected annotation path under data/: {ann_file}")
    return ann_file[len("data/"):]


def find_dataset_cfg(cfg: Config | ConfigDict) -> ConfigDict | None:
    current: Any = cfg
    seen = set()
    while isinstance(current, (Config, ConfigDict, dict)):
        current_id = id(current)
        if current_id in seen:
            return None
        seen.add(current_id)
        if "ann_file" in current:
            return current
        if "dataset" not in current:
            return None
        current = current["dataset"]
    return None


def expression_count(ann_file: Path) -> int:
    records = json.loads(ann_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{ann_file} must contain a JSON list.")
    count = 0
    for item in records:
        captions = item.get("captions")
        if not captions:
            captions = item.get("sentences")
        if not captions:
            captions = [item.get("caption") or item.get("text")]
        count += sum(1 for value in captions if value)
    return count


def configure_metric(
    metric_cfg: Any,
    ann_file: str,
    expected_count: int,
    results_dir: Path,
) -> None:
    if isinstance(metric_cfg, (ConfigDict, dict)):
        if "ann_file" in metric_cfg:
            metric_cfg["ann_file"] = ann_file
            metric_cfg["expected_num_expressions"] = expected_count
            metric_cfg["dump_results"] = True
            metric_cfg["dump_topk"] = 10
            metric_cfg["results_dir"] = str(results_dir)
        if "metrics" in metric_cfg:
            configure_metric(
                metric_cfg["metrics"], ann_file, expected_count, results_dir)
    elif isinstance(metric_cfg, list):
        for item in metric_cfg:
            configure_metric(item, ann_file, expected_count, results_dir)


def materialize_eval_config(
    base_config: Path,
    split: SplitSpec,
    output_dir: Path,
) -> tuple[Path, int]:
    cfg = Config.fromfile(str(base_config))
    expected_count = expression_count(repo_path(split.ann_file))

    for loader_key in ("val_dataloader", "test_dataloader"):
        if loader_key not in cfg:
            continue
        dataset_cfg = find_dataset_cfg(cfg[loader_key])
        if dataset_cfg is None:
            raise RuntimeError(f"{base_config} has no ann_file in {loader_key}.")
        dataset_cfg["ann_file"] = rel_ann_file(split.ann_file)
        dataset_cfg["data_root"] = "data/"
        dataset_cfg["img_prefix"] = "data/coco/train2014/"
        dataset_cfg["test_mode"] = True
        dataset_cfg["type"] = "CustomRefCOCOExpressionDataset"

    detailed_dir = output_dir / "formal_eval" / split.key
    for evaluator_key in ("val_evaluator", "test_evaluator"):
        if evaluator_key in cfg:
            configure_metric(
                cfg[evaluator_key],
                split.ann_file,
                expected_count,
                detailed_dir,
            )

    cfg.work_dir = str(output_dir / "work_dirs" / split.key)
    config_dir = output_dir / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    generated = config_dir / f"{split.key}.py"
    cfg.dump(str(generated))
    return generated, expected_count


def run_accuracy(
    eval_config: Path,
    checkpoint: str,
    split: SplitSpec,
    expected_count: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    work_dir = output_dir / "work_dirs" / split.key
    cmd = [
        "./tools/dist_test.sh",
        str(eval_config),
        checkpoint,
        str(args.gpus),
        "--work-dir",
        str(work_dir),
    ]
    shell_cmd = " ".join(shlex.quote(part) for part in cmd)
    log_file = output_dir / "logs" / f"{split.key}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return {
            "command": shell_cmd,
            "returncode": None,
            "status": "dry_run",
            "evaluation_unit": "expression",
            "expected_count": expected_count,
            "expected_num_expressions": expected_count,
            "log_file": str(log_file),
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_sec = time.time() - started
    log_file.write_text(proc.stdout, encoding="utf-8")
    acc_matches = ACC_PATTERN.findall(proc.stdout)
    count_matches = COUNT_PATTERN.findall(proc.stdout)
    accuracy = float(acc_matches[-1]) if acc_matches else None
    evaluated_count = int(count_matches[-1]) if count_matches else None
    count_matches_expected = evaluated_count == expected_count
    status = (
        "ok"
        if proc.returncode == 0
        and accuracy is not None
        and count_matches_expected
        else "failed"
    )
    return {
        "command": shell_cmd,
        "returncode": proc.returncode,
        "status": status,
        "evaluation_unit": "expression",
        "elapsed_sec": round(elapsed_sec, 3),
        "log_file": str(log_file),
        "stdout_tail": proc.stdout[-4000:],
        "acc_at_05": accuracy,
        "expected_count": expected_count,
        "evaluated_count": evaluated_count,
        "expected_num_expressions": expected_count,
        "evaluated_num_expressions": evaluated_count,
        "count_matches_expected": count_matches_expected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one config/checkpoint on all formal RefCOCO splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/refcoco_suite_accuracy")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = repo_path(args.config)
    checkpoint = repo_path(args.checkpoint)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(path)
        for path in (
            base_config,
            checkpoint,
            *(repo_path(split.ann_file) for split in SPLITS),
        )
        if not path.exists()
    ]
    if missing:
        payload = {
            "status": "failed",
            "failure_reasons": [f"missing input: {path}" for path in missing],
            "config": args.config,
            "checkpoint": args.checkpoint,
            "results": [],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    results = []
    for split in SPLITS:
        eval_config, expected_count = materialize_eval_config(
            base_config, split, output_dir)
        results.append({
            **asdict(split),
            "eval_config": str(eval_config),
            "ann_file": split.ann_file,
            "accuracy": run_accuracy(
                eval_config,
                args.checkpoint,
                split,
                expected_count,
                args,
                output_dir,
            ),
        })

    failed_splits = [
        item["key"]
        for item in results
        if item["accuracy"]["status"] == "failed"
    ]
    status = "dry_run" if args.dry_run else ("ok" if not failed_splits else "failed")
    summary = {
        "status": status,
        "failure_reasons": [
            f"formal evaluation failed: {split}" for split in failed_splits
        ],
        "config": args.config,
        "checkpoint": args.checkpoint,
        "gpus": args.gpus,
        "dry_run": args.dry_run,
        "splits": [asdict(split) for split in SPLITS],
        "protocols": {
            "formal_all_expressions": {
                "evaluation_unit": "expression",
                "expected_counts": {
                    item["key"]: item["accuracy"]["expected_num_expressions"]
                    for item in results
                },
            },
        },
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if status in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
