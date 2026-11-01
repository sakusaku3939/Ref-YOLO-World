"""Measure the complete end-to-end inference speed of Ref-YOLO-World.

Timing with `time.perf_counter()` alone underestimates real GPU time, because
CUDA kernels run asynchronously and only the CPU dispatch gets observed. This
script times one region per referring expression, starting immediately before
the image file is read and ending after the IoU computation and
`torch.cuda.synchronize()`. Model construction, checkpoint loading, and cache
pre-generation are excluded.

Per-stage timings for the backbone, neck, head, and text encoder come from
`tools/speed/stage_speed_breakdown.py`.
"""

import argparse
import json
import os
import os.path as osp
import statistics
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import torch
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import Compose
from mmyolo.utils import register_all_modules


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


def _stats(samples: List[float]) -> Dict[str, float | int]:
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


def _top_prediction(result: Any) -> Tuple[List[float], float]:
    pred = result.pred_instances if hasattr(result, "pred_instances") else result
    if len(pred.scores) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    top_idx = int(pred.scores.argmax())
    return (
        [float(x) for x in pred.bboxes[top_idx].detach().cpu().tolist()],
        float(pred.scores[top_idx]),
    )


def _compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def _cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"GPU idle check requires a CUDA device, got: {device}")


def _check_gpu_idle(device: str, max_util: float,
                    max_memory_mb: float) -> Dict[str, float | int | str]:
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
            f"Failed to query GPU state with nvidia-smi: {exc.output}") from exc

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


def _build_batch(test_pipeline: Compose, ann_file: str, item: Dict[str, Any],
                 idx: int) -> tuple[dict, list[list[str]]]:
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
        "img_id": item.get("_ref_index", idx),
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


def _set_text_state(model: Any, texts: List[List[str]],
                    text_cache: Dict[str, Any] | None) -> None:
    if text_cache is None:
        model.reparameterize(texts)
        return
    model.texts = texts
    model.text_feats, model.text_masks = text_cache[texts[0][0]]


def _expand_expression_items(
    ann_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand grouped RefCOCO annotations to one item per expression."""
    expression_items: List[Dict[str, Any]] = []
    for ref_index, item in enumerate(ann_items):
        captions = [
            str(caption).strip()
            for caption in item.get("captions", [])
            if str(caption).strip()
        ]
        for expression_index, caption in enumerate(captions):
            expression_item = deepcopy(item)
            expression_item["captions"] = [caption]
            expression_item["_ref_index"] = ref_index
            expression_item["_expression_index"] = expression_index
            expression_item["_expression_id"] = len(expression_items)
            expression_items.append(expression_item)
    return expression_items


def _select_items(
    ann_items: List[Dict[str, Any]],
    warmup: int,
    measure: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    warmup_items = ann_items[:warmup]
    if measure.lower() == "all":
        return warmup_items, ann_items, {
            "validation_total": len(ann_items),
            "measure_requested": measure,
            "measure_selection": "all_validation",
            "warmup_index_range": [0, len(warmup_items) - 1]
            if warmup_items else [],
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
        "warmup_index_range": [0, len(warmup_items) - 1]
        if warmup_items else [],
        "measured_index_range": [start, end - 1] if measured_items else [],
        "warmup_is_separate_from_measure": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure complete end-to-end speed from image loading to IoU.")
    parser.add_argument("--label", required=True, help="Arbitrary name recorded in the results")
    parser.add_argument("--config", required=True, help="Config to evaluate")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--ann-file", default="data/refcoco/refcocog/validation.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable the detector's inference_autocast during timing",
    )
    parser.add_argument(
        "--amp-dtype",
        default=None,
        help="Override the autocast dtype, e.g. float16 / bfloat16",
    )
    parser.add_argument(
        "--amp-scope",
        default=None,
        help="Override the autocast scope: both / forward / reparameterize",
    )
    parser.add_argument(
        "--amp-fp32-modules",
        nargs="+",
        default=None,
        help="Module aliases to keep in FP32 under AMP",
    )
    parser.add_argument(
        "--measure",
        default="500",
        help="Number of samples to measure: an integer, or 'all' for the whole validation set",
    )
    parser.add_argument(
        "--quick-500",
        action="store_true",
        help=(
            "Measure only 500 samples after warmup, for quick checks. "
            "Alias for --measure 500."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed passes; the reported FPS is the mean over passes.",
    )
    parser.add_argument(
        "--cache-text-features",
        action="store_true",
        help=(
            "Encode every expression once before timing and reuse the features "
            "(Stream inference for YOLO-World). Ref-YOLO-World sets its caches "
            "in the config instead."
        ),
    )
    parser.add_argument(
        "--skip-gpu-idle-check",
        action="store_true",
        help="Skip the GPU idle check via nvidia-smi before measurement",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=float,
        default=5.0,
        help="Maximum GPU utilization (%%) to treat the GPU as idle",
    )
    parser.add_argument(
        "--max-gpu-memory-mb",
        type=float,
        default=1024.0,
        help="Maximum GPU memory usage (MiB) to treat the GPU as idle",
    )
    parser.add_argument(
        "--prewarm-attention-reparameterization-cache",
        "--prewarm-reparam-cache",
        dest="prewarm_attention_reparameterization_cache",
        action="store_true",
        default=None,
        help=(
            "Run reparameterize on the measured texts beforehand to measure "
            "latency on attention reparameterization cache hits"
        ),
    )
    parser.add_argument(
        "--no-prewarm-attention-reparameterization-cache",
        dest="prewarm_attention_reparameterization_cache",
        action="store_false",
        help="Explicitly disable the prewarm setting in the config",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1.")
    if args.quick_500:
        if args.measure != "500":
            parser.error("--quick-500 cannot be combined with --measure other than 500.")
        args.measure = "500"
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires a CUDA-enabled runtime.")

    torch.backends.cudnn.benchmark = True
    register_all_modules()
    gpu_idle_check = None
    if not args.skip_gpu_idle_check:
        gpu_idle_check = _check_gpu_idle(
            args.device,
            args.max_gpu_util,
            args.max_gpu_memory_mb,
        )

    cfg = Config.fromfile(args.config)
    speed_measure_cfg = cfg.get("speed_measure", {})
    if args.prewarm_attention_reparameterization_cache is None:
        prewarm_attention_reparameterization_cache = bool(
            speed_measure_cfg.get(
                "prewarm_attention_reparameterization_cache", False))
    else:
        prewarm_attention_reparameterization_cache = bool(
            args.prewarm_attention_reparameterization_cache)
    cfg.work_dir = osp.join("/tmp", f"reliable_speed_{args.label}")
    test_pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    if any((
        args.amp,
        args.amp_dtype is not None,
        args.amp_scope is not None,
        args.amp_fp32_modules is not None,
    )):
        if not hasattr(model, "set_inference_autocast"):
            raise RuntimeError(
                "AMP overrides were requested, but the detector does not "
                "implement set_inference_autocast()."
            )
        model.set_inference_autocast(
            enabled=True,
            dtype=args.amp_dtype,
            scope=args.amp_scope,
            fp32_modules=args.amp_fp32_modules,
        )

    with open(args.ann_file, "r", encoding="utf-8") as f:
        grouped_ann_items = json.load(f)
    ann_items = _expand_expression_items(grouped_ann_items)
    if not ann_items:
        raise ValueError(
            "No referring expressions found in the annotation file.")
    warmup_items, measured_items, selection = _select_items(
        ann_items,
        args.warmup,
        args.measure,
    )

    complete_e2e_ms: List[float] = []
    fps_per_pass: List[float] = []
    correct = 0
    no_prediction = 0

    with torch.no_grad():
        text_cache = None
        if args.cache_text_features:
            text_cache = {
                item["captions"][0]: model.backbone.forward_text(
                    [[item["captions"][0]]])
                for item in warmup_items + measured_items
            }
            torch.cuda.synchronize()

        for idx, item in enumerate(warmup_items):
            batch, texts = _build_batch(test_pipeline, args.ann_file, item, idx)
            _set_text_state(model, texts, text_cache)
            _ = model.test_step(batch)[0]
            torch.cuda.synchronize()

        if prewarm_attention_reparameterization_cache:
            for measured_idx, item in enumerate(measured_items):
                _, texts = _build_batch(
                    test_pipeline,
                    args.ann_file,
                    item,
                    measured_idx,
                )
                model.reparameterize(texts)
            torch.cuda.synchronize()

        for pass_idx in range(args.repeat):
            pass_ms: List[float] = []
            for measured_idx, item in enumerate(measured_items):
                gt_bbox = item.get("bbox")
                if not isinstance(gt_bbox, list) or len(gt_bbox) != 4:
                    raise ValueError(
                        f"Annotation item {measured_idx} has invalid bbox: {gt_bbox}")

                torch.cuda.synchronize()
                started = time.perf_counter()
                batch, texts = _build_batch(
                    test_pipeline,
                    args.ann_file,
                    item,
                    measured_idx,
                )
                _set_text_state(model, texts, text_cache)
                result_sample = model.test_step(batch)[0]
                pred_bbox, score = _top_prediction(result_sample)
                iou = _compute_iou(pred_bbox, gt_bbox)
                torch.cuda.synchronize()
                pass_ms.append((time.perf_counter() - started) * 1000.0)

                if pass_idx == 0:
                    if score == 0.0 and pred_bbox == [0.0, 0.0, 0.0, 0.0]:
                        no_prediction += 1
                    elif iou >= 0.5:
                        correct += 1

                measured = measured_idx + 1
                if measured % 100 == 0 or measured == len(measured_items):
                    print(
                        f"{args.label} pass {pass_idx + 1}/{args.repeat} "
                        f"progress {measured}/{len(measured_items)} "
                        f"complete_e2e_mean_ms={statistics.fmean(pass_ms):.4f}",
                        flush=True,
                    )
            complete_e2e_ms.extend(pass_ms)
            fps_per_pass.append(1000.0 / statistics.fmean(pass_ms))

    latency = _stats(complete_e2e_ms)
    measured_total = len(measured_items)
    result = {
        "label": args.label,
        "warmup": len(warmup_items),
        "measure": len(measured_items),
        "quick_500": args.quick_500,
        "prewarm_attention_reparameterization_cache": prewarm_attention_reparameterization_cache,
        "prewarm_reparam_cache": prewarm_attention_reparameterization_cache,
        "selection": selection,
        "annotation_protocol": "one_sample_per_expression",
        "grouped_annotation_total": len(grouped_ann_items),
        "batch_size": 1,
        "correct_at_0_5": correct,
        "acc_at_0_5": round(correct / measured_total, 6) if measured_total else 0.0,
        "no_prediction": no_prediction,
        "complete_e2e_ms": latency,
        "complete_e2e_total_seconds": round(sum(complete_e2e_ms) / 1000.0, 4),
        "repeat": args.repeat,
        "complete_e2e_fps": round(statistics.fmean(fps_per_pass), 4),
        "complete_e2e_fps_per_pass": [round(fps, 4) for fps in fps_per_pass],
        "fps_formula": "mean over passes of 1000 / (mean complete_e2e_ms in the pass)",
        "cache_text_features": args.cache_text_features,
        "timing_scope": (
            "Per referring expression, timing starts immediately before the "
            "image file is read and includes decoding, the test pipeline, "
            "batch construction, reparameterization, the host-to-device copy, "
            "inference, NMS and postprocessing, the device-to-host copy of the "
            "prediction, the IoU computation, and CUDA synchronization. Model "
            "construction, checkpoint loading, and cache pre-generation are "
            "excluded."
        ),
        "gpu_idle_check": gpu_idle_check,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "amp": {
            "enabled": bool(args.amp),
            "dtype_override": args.amp_dtype,
            "scope_override": args.amp_scope,
            "fp32_modules_override": args.amp_fp32_modules,
        },
    }

    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
