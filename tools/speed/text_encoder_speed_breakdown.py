from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import statistics
import subprocess
import time
from typing import Any

import torch
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import Compose
from mmyolo.utils import register_all_modules


DEFAULT_YOLO_CONFIG = "configs/yolo_world/finetune_refcocog.py"
DEFAULT_YOLO_CHECKPOINT = (
    "weights/best_refcoco_yolo_world_v2_x_finetune_refcocog_epoch_14.pth"
)
DEFAULT_REF_CONFIG = (
    "configs/ref_yolo_world/ref_yolo_world_multidataset.py"
)
DEFAULT_REF_CHECKPOINT = (
    "weights/ref_yolo_world_x_multidataset.pth"
)
DEFAULT_ANN_FILE = "data/refcoco/refcocog/validation.json"


def _resolve_image_path(image_path: str, ann_file: str) -> str:
    if osp.isabs(image_path) and osp.exists(image_path):
        return image_path

    data_root = osp.dirname(osp.dirname(osp.dirname(ann_file)))
    candidates = [
        image_path,
        osp.join("data", image_path),
        osp.join(data_root, image_path),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return image_path


def _stats(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}

    sorted_samples = sorted(samples)
    p95_index = min(
        len(sorted_samples) - 1,
        max(0, int(round(0.95 * (len(sorted_samples) - 1)))),
    )
    return {
        "mean": round(sum(sorted_samples) / len(sorted_samples), 4),
        "p50": round(statistics.median(sorted_samples), 4),
        "p95": round(sorted_samples[p95_index], 4),
        "n": len(sorted_samples),
    }


def _cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"GPU idle check requires a CUDA device, got: {device}")


def _check_gpu_idle(device: str, max_util: float, max_memory_mb: float) -> dict[str, Any]:
    gpu_index = _cuda_device_index(device)
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            cmd,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is required for GPU idle check.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to query GPU state with nvidia-smi: {exc.output}"
        ) from exc

    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {first_line}")

    index, name, util, memory_used, memory_total = parts
    util_pct = float(util)
    memory_used_mb = float(memory_used)
    memory_total_mb = float(memory_total)
    is_idle = util_pct <= max_util and memory_used_mb <= max_memory_mb
    info = {
        "index": int(index),
        "name": name,
        "utilization_gpu_pct": util_pct,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "max_utilization_gpu_pct": max_util,
        "max_memory_used_mb": max_memory_mb,
        "is_idle": is_idle,
    }
    if not is_idle:
        raise RuntimeError(
            "Target GPU is not idle: "
            f"util={util_pct:.1f}% (max {max_util:.1f}%), "
            f"memory={memory_used_mb:.0f} MiB (max {max_memory_mb:.0f} MiB)."
        )
    return info


def _build_batch(
    test_pipeline: Compose,
    ann_file: str,
    item: dict[str, Any],
    idx: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    captions = [str(text).strip() for text in (item.get("captions") or []) if str(text).strip()]
    if not captions:
        captions = [
            str(sentence.get("sent", sentence.get("raw", ""))).strip()
            for sentence in (item.get("sentences") or [])
            if str(sentence.get("sent", sentence.get("raw", ""))).strip()
        ]
    if not captions:
        raise ValueError(f"Annotation item {idx} has no non-empty captions.")

    expr_ids = item.get("sent_ids")
    if not expr_ids:
        expr_ids = [
            sentence.get("sent_id", sentence.get("sentId"))
            for sentence in (item.get("sentences") or [])
        ]

    texts = [captions]
    data_info = {
        "img_id": idx,
        "ann_id": item.get("ann_id", item.get("ref_id", idx)),
        "expr_ids": expr_ids,
        "img_path": _resolve_image_path(item["image_path"], ann_file),
        "texts": texts,
    }
    data_info = test_pipeline(data_info)
    batch = {
        "inputs": data_info["inputs"].unsqueeze(0),
        "data_samples": [data_info["data_samples"]],
    }
    return batch, texts


def _select_items(
    ann_items: list[dict[str, Any]],
    warmup: int,
    measure: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    warmup_items = ann_items[:warmup]
    if measure.lower() == "all":
        return warmup_items, ann_items, {
            "validation_total": len(ann_items),
            "measure_requested": measure,
            "measure_selection": "all_validation",
            "warmup_index_range": [0, len(warmup_items) - 1] if warmup_items else [],
            "measured_index_range": [0, len(ann_items) - 1] if ann_items else [],
            "warmup_is_separate_from_measure": True,
        }

    try:
        measure_count = int(measure)
    except ValueError as exc:
        raise ValueError("--measure must be an integer or 'all'.") from exc
    if measure_count < 0:
        raise ValueError("--measure must be non-negative.")

    start = warmup
    end = min(len(ann_items), warmup + measure_count)
    measured_items = ann_items[start:end]
    return warmup_items, measured_items, {
        "validation_total": len(ann_items),
        "measure_requested": measure_count,
        "measure_selection": "contiguous_after_warmup",
        "warmup_index_range": [0, len(warmup_items) - 1] if warmup_items else [],
        "measured_index_range": [start, end - 1] if measured_items else [],
        "warmup_is_separate_from_measure": False,
    }


def _load_model(config_path: str, checkpoint_path: str, device: str, label: str):
    cfg = Config.fromfile(config_path)
    cfg.work_dir = osp.join("/tmp", f"text_encoder_speed_breakdown_{label}")
    test_pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))
    model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
    model.eval()
    if not hasattr(model, "reparameterize"):
        raise RuntimeError(f"{label} model does not expose reparameterize(texts).")
    return model, test_pipeline


def _measure_model(
    *,
    label: str,
    config_path: str,
    checkpoint_path: str,
    ann_file: str,
    warmup_items: list[dict[str, Any]],
    measured_items: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    model, test_pipeline = _load_model(config_path, checkpoint_path, device, label)

    preprocess_ms: list[float] = []
    text_encoder_ms: list[float] = []
    forward_sync_ms: list[float] = []
    forward_event_ms: list[float] = []
    e2e_model_ms: list[float] = []

    with torch.no_grad():
        for idx, item in enumerate(warmup_items):
            batch, texts = _build_batch(test_pipeline, ann_file, item, idx)
            model.reparameterize(texts)
            _ = model.test_step(batch)[0]
            torch.cuda.synchronize()

        for measured_idx, item in enumerate(measured_items):
            preprocess_start = time.perf_counter()
            batch, texts = _build_batch(test_pipeline, ann_file, item, measured_idx)
            preprocess_end = time.perf_counter()
            preprocess_ms.append((preprocess_end - preprocess_start) * 1000.0)

            torch.cuda.synchronize()
            text_start = time.perf_counter()
            model.reparameterize(texts)
            torch.cuda.synchronize()
            text_end = time.perf_counter()

            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            forward_sync_start = time.perf_counter()
            forward_start_event.record()
            _ = model.test_step(batch)[0]
            forward_end_event.record()
            torch.cuda.synchronize()
            forward_sync_end = time.perf_counter()

            torch.cuda.synchronize()
            e2e_start = time.perf_counter()
            model.reparameterize(texts)
            _ = model.test_step(batch)[0]
            torch.cuda.synchronize()
            e2e_end = time.perf_counter()

            text_encoder_ms.append((text_end - text_start) * 1000.0)
            forward_sync_ms.append((forward_sync_end - forward_sync_start) * 1000.0)
            forward_event_ms.append(forward_start_event.elapsed_time(forward_end_event))
            e2e_model_ms.append((e2e_end - e2e_start) * 1000.0)

            measured = measured_idx + 1
            if measured % 100 == 0 or measured == len(measured_items):
                text_mean = sum(text_encoder_ms) / len(text_encoder_ms)
                forward_mean = sum(forward_sync_ms) / len(forward_sync_ms)
                e2e_mean = sum(e2e_model_ms) / len(e2e_model_ms)
                print(
                    f"{label} progress {measured}/{len(measured_items)} "
                    f"text_ms={text_mean:.2f} "
                    f"forward_no_text_ms={forward_mean:.2f} "
                    f"e2e_model_ms={e2e_mean:.2f}",
                    flush=True,
                )

    result = {
        "label": label,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "preprocess_ms": _stats(preprocess_ms),
        "text_encoder_ms": _stats(text_encoder_ms),
        "forward_without_text_encoder_sync_ms": _stats(forward_sync_ms),
        "forward_without_text_encoder_event_ms": _stats(forward_event_ms),
        "e2e_model_ms": _stats(e2e_model_ms),
    }
    e2e_mean = result["e2e_model_ms"]["mean"]
    result["e2e_fps"] = round(1000.0 / e2e_mean, 4) if e2e_mean > 0 else 0.0

    del model
    del test_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _row(model_name: str, result: dict[str, Any], yolo_e2e_mean: float | None) -> dict[str, Any]:
    e2e_mean = float(result["e2e_model_ms"]["mean"])
    delta = None if yolo_e2e_mean is None else round(e2e_mean - yolo_e2e_mean, 4)
    return {
        "model": model_name,
        "preprocess_ms": result["preprocess_ms"]["mean"],
        "text_encoder_ms": result["text_encoder_ms"]["mean"],
        "forward_without_text_encoder_ms": result[
            "forward_without_text_encoder_sync_ms"
        ]["mean"],
        "e2e_model_ms": e2e_mean,
        "e2e_fps": result["e2e_fps"],
        "e2e_delta_ms_vs_yolo_world": delta,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _make_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Preprocess ms",
        "Text encoder ms",
        "Forward without text encoder ms",
        "E2E model ms",
        "E2E FPS",
        "E2E delta vs YOLO-World ms",
    ]
    keys = [
        "model",
        "preprocess_ms",
        "text_encoder_ms",
        "forward_without_text_encoder_ms",
        "e2e_model_ms",
        "e2e_fps",
        "e2e_delta_ms_vs_yolo_world",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[key]) for key in keys) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: str,
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    gpu_idle_check: dict[str, Any] | None,
) -> str:
    table = _make_table(rows)
    report = "\n".join(
        [
            "# Text Encoder Speed Breakdown",
            "",
            table,
            "",
            "## Notes",
            "",
            "- `Text encoder ms` measures `model.reparameterize(texts)` with CUDA synchronization.",
            "- `Forward without text encoder ms` measures `model.test_step(batch)` after reparameterization.",
            "- `E2E model ms` measures `reparameterize(texts) + test_step(batch)` after preprocessing.",
            "- `Preprocess ms` measures test pipeline construction on the host and is not included in `E2E model ms`.",
            "",
            "## Selection",
            "",
            "```json",
            json.dumps(selection, indent=2, ensure_ascii=False),
            "```",
            "",
            "## GPU Idle Check",
            "",
            "```json",
            json.dumps(gpu_idle_check, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    report_path = osp.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure text encoder and no-text-forward latency for YOLO-World "
            "and Ref-YOLO-World under the same RefCOCOg subset."
        )
    )
    parser.add_argument("--yolo-config", default=DEFAULT_YOLO_CONFIG)
    parser.add_argument("--yolo-checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--ref-config", default=DEFAULT_REF_CONFIG)
    parser.add_argument("--ref-checkpoint", default=DEFAULT_REF_CHECKPOINT)
    parser.add_argument("--ann-file", default=DEFAULT_ANN_FILE)
    parser.add_argument("--output-dir", default="results/text_encoder_speed_breakdown")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--measure",
        default="500",
        help="Number of measured validation items, or 'all'.",
    )
    parser.add_argument("--skip-gpu-idle-check", action="store_true")
    parser.add_argument("--max-gpu-util", type=float, default=5.0)
    parser.add_argument("--max-gpu-memory-mb", type=float, default=1024.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires a CUDA-enabled runtime.")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    register_all_modules()

    gpu_idle_check = None
    if not args.skip_gpu_idle_check:
        gpu_idle_check = _check_gpu_idle(
            args.device,
            args.max_gpu_util,
            args.max_gpu_memory_mb,
        )

    with open(args.ann_file, "r", encoding="utf-8") as f:
        ann_items = json.load(f)
    warmup_items, measured_items, selection = _select_items(
        ann_items,
        args.warmup,
        args.measure,
    )
    if not measured_items:
        raise RuntimeError("No measured items were selected.")

    yolo_result = _measure_model(
        label="yolo_world",
        config_path=args.yolo_config,
        checkpoint_path=args.yolo_checkpoint,
        ann_file=args.ann_file,
        warmup_items=warmup_items,
        measured_items=measured_items,
        device=args.device,
    )
    ref_result = _measure_model(
        label="ref_yolo_world",
        config_path=args.ref_config,
        checkpoint_path=args.ref_checkpoint,
        ann_file=args.ann_file,
        warmup_items=warmup_items,
        measured_items=measured_items,
        device=args.device,
    )

    yolo_e2e_mean = float(yolo_result["e2e_model_ms"]["mean"])
    rows = [
        _row("YOLO-World", yolo_result, None),
        _row("Ref-YOLO-World", ref_result, yolo_e2e_mean),
    ]
    table = _write_report(args.output_dir, rows, selection, gpu_idle_check)

    summary = {
        "overall_pass": all(row["e2e_model_ms"] > 0 for row in rows),
        "comparison_table": rows,
        "comparison_table_markdown": table,
        "selection": selection,
        "gpu_idle_check": gpu_idle_check,
        "models": {
            "yolo_world": yolo_result,
            "ref_yolo_world": ref_result,
        },
        "notes": [
            "Text encoder timing is model.reparameterize(texts).",
            "Forward without text encoder timing is model.test_step(batch) after reparameterization.",
            "E2E model timing excludes host preprocessing and includes reparameterize + test_step.",
        ],
    }
    summary_path = osp.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(table, flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
