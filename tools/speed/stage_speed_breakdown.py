from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import statistics
import subprocess
import time
from collections import defaultdict
from typing import Any, Callable

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
    "work_dirs/"
    "yolo_world_v2_x_finetune_refcocog_token_aware_pe_crossattn_multi_dataset_spatial_flip_text_lora/"
    "yolo_world_v2_x_finetune_refcocog_token_aware_pe_crossattn_multi_dataset_spatial_flip_text_lora.py"
)
DEFAULT_REF_CHECKPOINT = (
    "weights/ref_yolo_world_x_multidataset_text_lora.pth"
)
DEFAULT_ANN_FILE = "data/refcoco/refcocog/validation.json"


class StageProfiler:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.current: dict[str, float] = defaultdict(float)
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []
        backbone_target = getattr(model.backbone, "image_model", model.backbone)
        self._patches: list[tuple[str, Any, str]] = [
            ("model_data_preprocessor_ms", model.data_preprocessor, "forward"),
            ("backbone_ms", backbone_target, "forward"),
            ("neck_ms", model.neck, "forward"),
            ("head_ms", model.bbox_head, "forward"),
            ("postprocess_predict_by_feat_ms", model.bbox_head, "predict_by_feat"),
            ("postprocess_add_datasample_ms", model, "add_pred_to_datasample"),
        ]

    def __enter__(self):
        for stage_name, obj, attr in self._patches:
            if not hasattr(obj, attr):
                continue
            original = getattr(obj, attr)
            self._originals.append((obj, attr, original))

            def wrapped(*args, _original=original, _stage_name=stage_name, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = _original(*args, **kwargs)
                torch.cuda.synchronize()
                self.current[_stage_name] += (time.perf_counter() - start) * 1000.0
                return out

            setattr(obj, attr, wrapped)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for obj, attr, original in reversed(self._originals):
            setattr(obj, attr, original)
        self._originals.clear()

    def reset(self) -> None:
        self.current = defaultdict(float)

    def snapshot(self) -> dict[str, float]:
        return dict(self.current)


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
        output = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.STDOUT).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is required for GPU idle check.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to query GPU state with nvidia-smi: {exc.output}") from exc

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
    cfg.work_dir = osp.join("/tmp", f"stage_speed_breakdown_{label}")
    test_pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))
    model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
    model.eval()
    if not hasattr(model, "reparameterize"):
        raise RuntimeError(f"{label} model does not expose reparameterize(texts).")
    return model, test_pipeline


def _append(samples: dict[str, list[float]], key: str, value: float) -> None:
    samples.setdefault(key, []).append(float(value))


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
    samples: dict[str, list[float]] = {}

    with torch.no_grad():
        for idx, item in enumerate(warmup_items):
            batch, texts = _build_batch(test_pipeline, ann_file, item, idx)
            model.reparameterize(texts)
            _ = model.test_step(batch)[0]
            torch.cuda.synchronize()

        with StageProfiler(model) as profiler:
            for measured_idx, item in enumerate(measured_items):
                pipeline_start = time.perf_counter()
                batch, texts = _build_batch(test_pipeline, ann_file, item, measured_idx)
                pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

                torch.cuda.synchronize()
                text_start = time.perf_counter()
                model.reparameterize(texts)
                torch.cuda.synchronize()
                text_ms = (time.perf_counter() - text_start) * 1000.0

                profiler.reset()
                torch.cuda.synchronize()
                test_step_start = time.perf_counter()
                _ = model.test_step(batch)[0]
                torch.cuda.synchronize()
                test_step_ms = (time.perf_counter() - test_step_start) * 1000.0
                stages = profiler.snapshot()

                data_preprocessor_ms = stages.get("model_data_preprocessor_ms", 0.0)
                preprocess_ms = pipeline_ms + data_preprocessor_ms
                backbone_ms = stages.get("backbone_ms", 0.0)
                neck_ms = stages.get("neck_ms", 0.0)
                head_ms = stages.get("head_ms", 0.0)
                postprocess_ms = (
                    stages.get("postprocess_predict_by_feat_ms", 0.0)
                    + stages.get("postprocess_add_datasample_ms", 0.0)
                )
                accounted_test_step_ms = (
                    data_preprocessor_ms + backbone_ms + neck_ms + head_ms + postprocess_ms
                )
                residual_ms = test_step_ms - accounted_test_step_ms
                model_total_ms = text_ms + test_step_ms
                total_with_preprocess_ms = pipeline_ms + model_total_ms
                total_without_preprocess_ms = model_total_ms - data_preprocessor_ms

                _append(samples, "pipeline_preprocess_ms", pipeline_ms)
                _append(samples, "model_data_preprocessor_ms", data_preprocessor_ms)
                _append(samples, "preprocess_ms", preprocess_ms)
                _append(samples, "text_encoder_ms", text_ms)
                _append(samples, "backbone_ms", backbone_ms)
                _append(samples, "neck_ms", neck_ms)
                _append(samples, "head_ms", head_ms)
                _append(samples, "postprocess_ms", postprocess_ms)
                _append(samples, "test_step_total_ms", test_step_ms)
                _append(samples, "accounted_test_step_ms", accounted_test_step_ms)
                _append(samples, "residual_ms", residual_ms)
                _append(samples, "model_total_ms", model_total_ms)
                _append(samples, "total_with_preprocess_ms", total_with_preprocess_ms)
                _append(samples, "total_without_preprocess_ms", total_without_preprocess_ms)

                measured = measured_idx + 1
                if measured % 100 == 0 or measured == len(measured_items):
                    print(
                        f"{label} progress {measured}/{len(measured_items)} "
                        f"total_no_preprocess={sum(samples['total_without_preprocess_ms']) / len(samples['total_without_preprocess_ms']):.2f} "
                        f"backbone={sum(samples['backbone_ms']) / len(samples['backbone_ms']):.2f} "
                        f"neck={sum(samples['neck_ms']) / len(samples['neck_ms']):.2f} "
                        f"head={sum(samples['head_ms']) / len(samples['head_ms']):.2f} "
                        f"post={sum(samples['postprocess_ms']) / len(samples['postprocess_ms']):.2f}",
                        flush=True,
                    )

    result = {
        "label": label,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "metrics": {key: _stats(value) for key, value in samples.items()},
    }
    total_mean = result["metrics"]["total_without_preprocess_ms"]["mean"]
    model_mean = result["metrics"]["model_total_ms"]["mean"]
    result["total_without_preprocess_fps"] = round(1000.0 / total_mean, 4) if total_mean > 0 else 0.0
    result["total_with_preprocess_fps"] = round(1000.0 / result["metrics"]["total_with_preprocess_ms"]["mean"], 4) if result["metrics"]["total_with_preprocess_ms"]["mean"] > 0 else 0.0
    result["model_fps"] = round(1000.0 / model_mean, 4) if model_mean > 0 else 0.0

    del model
    del test_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _row(model_name: str, result: dict[str, Any]) -> dict[str, Any]:
    m = result["metrics"]
    return {
        "model": model_name,
        "text_encoder_ms": m["text_encoder_ms"]["mean"],
        "backbone_ms": m["backbone_ms"]["mean"],
        "neck_ms": m["neck_ms"]["mean"],
        "head_ms": m["head_ms"]["mean"],
        "postprocess_ms": m["postprocess_ms"]["mean"],
        "residual_ms": m["residual_ms"]["mean"],
        "total_without_preprocess_ms": m["total_without_preprocess_ms"]["mean"],
        "fps": result["total_without_preprocess_fps"],
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _make_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "モデル",
        "テキストエンコーダー ms",
        "Backbone ms",
        "Neck ms",
        "Head ms",
        "後処理 ms",
        "残差 ms",
        "合計 ms",
        "FPS",
    ]
    keys = [
        "model",
        "text_encoder_ms",
        "backbone_ms",
        "neck_ms",
        "head_ms",
        "postprocess_ms",
        "residual_ms",
        "total_without_preprocess_ms",
        "fps",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[key]) for key in keys) + " |")
    return "\n".join(lines)


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 2:
        return []
    base, ref = rows
    delta = {"model": "Ref - YOLO"}
    for key, value in ref.items():
        if key == "model":
            continue
        base_value = base[key]
        if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
            delta[key] = round(float(value) - float(base_value), 4)
    return [delta]


def _write_report(output_dir: str, rows: list[dict[str, Any]], selection: dict[str, Any], gpu_idle_check: dict[str, Any] | None) -> tuple[str, str]:
    table = _make_table(rows)
    delta_table = _make_table(_delta_rows(rows))
    report = "\n".join(
        [
            "# Stage Speed Breakdown",
            "",
            "## YOLO-World vs Ref-YOLO-World",
            "",
            table,
            "",
            "## Delta",
            "",
            delta_table,
            "",
            "## Notes",
            "",
            "- 前処理は表と合計から除外する。raw metricsには `pipeline_preprocess_ms` と `model_data_preprocessor_ms` を残す。",
            "- `テキストエンコーダー ms` is `model.reparameterize(texts)`.",
            "- `Head ms` is `bbox_head.forward` only.",
            "- `後処理 ms` is `bbox_head.predict_by_feat` plus `add_pred_to_datasample`.",
            "- `残差 ms` is unaccounted test_step overhead after wrapped stages.",
            "- Stage wrapping adds CUDA synchronization around each stage, so use this for attribution rather than direct latency comparison with unwrapped E2E.",
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
    with open(osp.join(output_dir, "stage_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return table, delta_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure YOLO-World and Ref-YOLO-World stage latency.")
    parser.add_argument("--yolo-config", default=DEFAULT_YOLO_CONFIG)
    parser.add_argument("--yolo-checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--ref-config", default=DEFAULT_REF_CONFIG)
    parser.add_argument("--ref-checkpoint", default=DEFAULT_REF_CHECKPOINT)
    parser.add_argument("--ann-file", default=DEFAULT_ANN_FILE)
    parser.add_argument("--output-dir", default="results/stage_speed_breakdown")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", default="500", help="Number of measured validation items, or 'all'.")
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
        gpu_idle_check = _check_gpu_idle(args.device, args.max_gpu_util, args.max_gpu_memory_mb)

    with open(args.ann_file, "r", encoding="utf-8") as f:
        ann_items = json.load(f)
    warmup_items, measured_items, selection = _select_items(ann_items, args.warmup, args.measure)
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

    rows = [_row("YOLO-World", yolo_result), _row("Ref-YOLO-World", ref_result)]
    table, delta_table = _write_report(args.output_dir, rows, selection, gpu_idle_check)
    summary = {
        "overall_pass": all(row["total_without_preprocess_ms"] > 0 for row in rows),
        "stage_breakdown_table": rows,
        "stage_breakdown_table_markdown": table,
        "stage_delta_table": _delta_rows(rows),
        "stage_delta_table_markdown": delta_table,
        "selection": selection,
        "gpu_idle_check": gpu_idle_check,
        "models": {
            "yolo_world": yolo_result,
            "ref_yolo_world": ref_result,
        },
    }
    with open(osp.join(args.output_dir, "stage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(table, flush=True)
    print("\nDelta\n", flush=True)
    print(delta_table, flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
