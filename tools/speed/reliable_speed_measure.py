"""YOLO-World / Ref-YOLO-World の推論速度を同期付きで計測する。

従来の `time.perf_counter()` だけの計測では CUDA kernel の非同期実行により
CPU dispatch 時間だけを測ってしまうため、実 GPU 時間を過小評価しやすい。
このスクリプトでは `torch.cuda.synchronize()` と CUDA Event の両方を使い、
reparameterize、YOLO-only、end-to-end の時間を同じ per-image loop で測る。
あわせて、measured images 全体を一括で測る throughput 指標も出す。
"""

import argparse
import json
import os
import os.path as osp
import statistics
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Tuple

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


def _throughput_stats(total_sec: float, total_images: int) -> Dict[str, float | int]:
    if total_images <= 0 or total_sec <= 0:
        return {
            "total_sec": 0.0,
            "total_images": total_images,
            "fps": 0.0,
        }
    return {
        "total_sec": round(total_sec, 4),
        "total_images": total_images,
        "fps": round(total_images / total_sec, 4),
    }


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
        description="CUDA synchronize / Event を使って推論速度を測る。")
    parser.add_argument("--label", required=True, help="結果に記録する任意の名前")
    parser.add_argument("--config", required=True, help="評価対象 config")
    parser.add_argument("--checkpoint", required=True, help="評価対象 checkpoint")
    parser.add_argument("--ann-file", default="data/refcoco/refcocog/validation.json")
    parser.add_argument("--output", required=True, help="JSON 出力先")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="推論時計測で detector の inference_autocast を有効化する",
    )
    parser.add_argument(
        "--amp-dtype",
        default=None,
        help="autocast dtype を上書きする。例: float16 / bfloat16",
    )
    parser.add_argument(
        "--amp-scope",
        default=None,
        help="autocast scope を上書きする。both / forward / reparameterize",
    )
    parser.add_argument(
        "--amp-fp32-modules",
        nargs="+",
        default=None,
        help="AMP 中も FP32 で実行する module alias 群",
    )
    parser.add_argument(
        "--measure",
        default="500",
        help="計測件数。整数、または validation 全件を測る 'all'",
    )
    parser.add_argument(
        "--quick-500",
        action="store_true",
        help=(
            "research-analyze などの短時間確認用に、warmup 後の 500 件だけを"
            "計測することを明示する。--measure 500 の別名。"
        ),
    )
    parser.add_argument(
        "--skip-gpu-idle-check",
        action="store_true",
        help="計測前の nvidia-smi による GPU idle check を省略する",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=float,
        default=5.0,
        help="idle とみなす最大 GPU 使用率 (%%)",
    )
    parser.add_argument(
        "--max-gpu-memory-mb",
        type=float,
        default=1024.0,
        help="idle とみなす最大 GPU メモリ使用量 (MiB)",
    )
    parser.add_argument(
        "--no-throughput",
        action="store_true",
        help="measured images 全体を一括で測る throughput 指標を省略する",
    )
    parser.add_argument(
        "--prewarm-attention-reparameterization-cache",
        "--prewarm-reparam-cache",
        dest="prewarm_attention_reparameterization_cache",
        action="store_true",
        default=None,
        help=(
            "計測対象 texts の reparameterize を事前実行し、"
            "attention reparameterization cache hit 時の latency を測る"
        ),
    )
    parser.add_argument(
        "--no-prewarm-attention-reparameterization-cache",
        dest="prewarm_attention_reparameterization_cache",
        action="store_false",
        help="config 側の prewarm 設定を明示的に無効化する",
    )
    args = parser.parse_args()
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

    reparam_sync_ms: List[float] = []
    yolo_sync_ms: List[float] = []
    yolo_event_ms: List[float] = []
    e2e_sync_ms: List[float] = []
    measured_batches: List[tuple[dict, list[list[str]]]] = []

    with torch.no_grad():
        for idx, item in enumerate(warmup_items):
            batch, texts = _build_batch(test_pipeline, args.ann_file, item, idx)
            model.reparameterize(texts)
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

        for measured_idx, item in enumerate(measured_items):
            batch, texts = _build_batch(
                test_pipeline,
                args.ann_file,
                item,
                measured_idx,
            )

            measured_batches.append((batch, texts))

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.reparameterize(texts)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.test_step(batch)[0]
            end_event.record()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            reparam_sync_ms.append((t1 - t0) * 1000.0)
            yolo_sync_ms.append((t2 - t1) * 1000.0)
            yolo_event_ms.append(start_event.elapsed_time(end_event))
            e2e_sync_ms.append((t2 - t0) * 1000.0)

            measured = measured_idx + 1
            if measured % 100 == 0 or measured == len(measured_items):
                mean_sync = sum(yolo_sync_ms) / len(yolo_sync_ms)
                mean_event = sum(yolo_event_ms) / len(yolo_event_ms)
                print(
                    f"{args.label} progress {measured}/{len(measured_items)} "
                    f"yolo_sync_mean={mean_sync:.2f} "
                    f"yolo_event_mean={mean_event:.2f}",
                    flush=True,
                )

        throughput_e2e = None
        if not args.no_throughput:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for batch, texts in measured_batches:
                model.reparameterize(texts)
                _ = model.test_step(batch)[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            throughput_e2e = _throughput_stats(
                t1 - t0,
                len(measured_batches),
            )

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
        "reparam_sync_ms": _stats(reparam_sync_ms),
        "yolo_sync_ms": _stats(yolo_sync_ms),
        "yolo_event_ms": _stats(yolo_event_ms),
        "e2e_sync_ms": _stats(e2e_sync_ms),
        "throughput_e2e": throughput_e2e,
        "throughput_note": (
            "throughput_e2e は measured images を事前に batch 化した後、"
            "reparameterize + test_step のループ全体を同期付きで一括計測する。"
            "FPS 計測用の指標であり、per-image latency ではない。"
            "データ読み込みと test pipeline 構築時間は含めない。"
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
