"""Evaluate ONNX raw outputs as final RefCOCOg bboxes.

This script closes the validation gap after export: it runs the text-cache ONNX
model, converts raw head outputs to final boxes with the existing
RefYOLOWorldHead postprocess, and computes top-1 Acc@0.5 on RefCOCOg val.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import statistics
import time
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX Runtime outputs through existing bbox postprocess and compute Acc@0.5.")
    parser.add_argument("config", help="Config with text cache settings")
    parser.add_argument("checkpoint", help="Checkpoint used to instantiate the Ref-YOLO-World model")
    parser.add_argument("--onnx", required=True, help="Text-cache-input ONNX model path")
    parser.add_argument("--ann-file", default="data/refcoco/refcocog/validation.json")
    parser.add_argument("--output", required=True, help="Summary JSON output path")
    parser.add_argument("--details-output", default=None, help="Per-sample detailed JSON output path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provider", default="CUDAExecutionProvider",
                        choices=("CUDAExecutionProvider", "CPUExecutionProvider"))
    parser.add_argument("--max-samples", default="all",
                        help="Number of validation samples to evaluate, or 'all'")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Exclude the first N evaluated samples from latency statistics only")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def mkdir_parent(path: str) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_annotations(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list annotations in {path}")
    return data


def max_samples_value(raw: str, total: int) -> int:
    if str(raw).lower() == "all":
        return total
    value = int(raw)
    if value < 0:
        return total
    return min(value, total)


def compute_iou_np(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def latency_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    mean_ms = sum(ordered) / len(ordered)
    return {
        "n": len(ordered),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[p95_idx], 4),
        "fps": round(1000.0 / mean_ms, 4) if mean_ms > 0 else 0.0,
    }


def build_runner_and_model(config_path: str, checkpoint_path: str, device: str) -> tuple[Runner, torch.nn.Module]:
    cfg = Config.fromfile(config_path)
    cfg.work_dir = osp.join("/tmp", "onnx_bbox_acc_eval")
    cfg.custom_hooks = []
    cfg.default_hooks = dict(logger=dict(type="LoggerHook"))
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    runner = Runner.from_cfg(cfg)
    model = runner.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    model = model.to(device)
    model.eval()
    return runner, model


def ort_session(onnx_path: str, provider: str) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    if provider not in available:
        raise RuntimeError(f"Requested provider {provider} is not available. Available: {sorted(available)}")
    return ort.InferenceSession(onnx_path, providers=[provider])


def expected_input_dim(session: ort.InferenceSession, name: str, axis: int) -> int | None:
    for item in session.get_inputs():
        if item.name == name and axis < len(item.shape):
            dim = item.shape[axis]
            return int(dim) if isinstance(dim, int) and dim > 0 else None
    return None


def fit_axis(tensor: torch.Tensor, axis: int, target: int | None) -> torch.Tensor:
    if target is None:
        return tensor
    current = int(tensor.shape[axis])
    if current == target:
        return tensor
    if current > target:
        index = [slice(None)] * tensor.ndim
        index[axis] = slice(0, target)
        return tensor[tuple(index)]

    pad_shape = list(tensor.shape)
    pad_shape[axis] = target - current
    padding = tensor.new_zeros(pad_shape)
    return torch.cat((tensor, padding), dim=axis)


def numpy_inputs(input_names: Sequence[str], tensors: Sequence[torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, tensors)}


def output_tensors(outputs: Sequence[np.ndarray], num_levels: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(outputs) != num_levels * 2:
        raise ValueError(f"Expected {num_levels * 2} ONNX outputs, got {len(outputs)}")
    tensors = [torch.from_numpy(output).to(device=device) for output in outputs]
    return tensors[:num_levels], tensors[num_levels:]


def sample_texts(data_samples: Sequence[Any]) -> list[list[str]]:
    rows = []
    for data_sample in data_samples:
        if not hasattr(data_sample, "texts"):
            raise AttributeError("data sample does not contain texts")
        rows.append(data_sample.texts)
    return rows


def sample_metas(data_samples: Sequence[Any]) -> list[dict[str, Any]]:
    return [dict(data_sample.metainfo) if hasattr(data_sample, "metainfo") else {} for data_sample in data_samples]


def top_prediction(result: Any) -> tuple[list[float], float]:
    pred = result.pred_instances if hasattr(result, "pred_instances") else result
    if len(pred.scores) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    top_idx = int(pred.scores.argmax())
    return [float(x) for x in pred.bboxes[top_idx].detach().cpu().tolist()], float(pred.scores[top_idx])


def annotation_text(item: dict[str, Any]) -> str:
    captions = item.get("captions") or []
    return str(captions[0]) if captions else ""


def main() -> None:
    args = parse_args()
    register_all_modules()
    annotations = load_annotations(args.ann_file)
    limit = max_samples_value(args.max_samples, len(annotations))
    device = torch.device(args.device)

    runner, model = build_runner_and_model(args.config, args.checkpoint, args.device)
    if getattr(model, "text_feature_cache", None) is None:
        raise RuntimeError("Config/model must enable text_feature_cache for ONNX text tensor inputs.")

    session = ort_session(args.onnx, args.provider)
    input_names = [item.name for item in session.get_inputs()]
    expected_text_slots = expected_input_dim(session, "pooled_text_features", 1)
    expected_token_len = expected_input_dim(session, "token_features", 2)
    num_levels = len(model.bbox_head.head_module.featmap_strides)
    details: list[dict[str, Any]] = []
    onnx_ms: list[float] = []
    preprocess_ms: list[float] = []
    text_cache_ms: list[float] = []
    input_pack_ms: list[float] = []
    output_tensor_ms: list[float] = []
    predict_by_feat_ms: list[float] = []
    postprocess_ms: list[float] = []
    metric_ms: list[float] = []
    e2e_ms: list[float] = []
    n_correct = 0
    n_empty = 0
    total_start = time.perf_counter()

    dataloader = runner.val_dataloader
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            if idx >= limit:
                break
            if idx % args.progress_interval == 0:
                print(f"eval {idx}/{limit}", flush=True)
            ann = annotations[idx]
            gt_bbox = ann.get("bbox")
            if not isinstance(gt_bbox, list) or len(gt_bbox) != 4:
                raise ValueError(f"Annotation index {idx} has invalid bbox: {gt_bbox}")

            e2e_start = time.perf_counter()
            preprocess_start = time.perf_counter()
            data = model.data_preprocessor(data_batch, False)
            inputs = data["inputs"]
            data_samples = data["data_samples"]
            texts = sample_texts(data_samples)
            preprocess_elapsed_ms = (time.perf_counter() - preprocess_start) * 1000.0

            text_cache_start = time.perf_counter()
            cached = model._lookup_text_cache(texts, data_samples, inputs)
            if cached is None:
                raise RuntimeError(f"Text cache lookup failed at sample {idx}")
            text_cache_elapsed_ms = (time.perf_counter() - text_cache_start) * 1000.0

            input_pack_start = time.perf_counter()
            pooled_text_features = fit_axis(cached.pooled_text_features, 1, expected_text_slots)
            token_features = fit_axis(cached.token_features, 1, expected_text_slots)
            token_features = fit_axis(token_features, 2, expected_token_len)

            text_mask = cached.text_mask
            if text_mask is None:
                text_mask = torch.ones(pooled_text_features.shape[:2], device=inputs.device, dtype=inputs.dtype)
            else:
                text_mask = fit_axis(text_mask, 1, expected_text_slots)
            token_mask = cached.token_attention_mask
            if token_mask is None:
                token_mask = torch.ones(token_features.shape[:3], device=inputs.device, dtype=inputs.dtype)
            else:
                token_mask = fit_axis(token_mask, 1, expected_text_slots)
                token_mask = fit_axis(token_mask, 2, expected_token_len)
            if text_mask.dtype != inputs.dtype:
                text_mask = text_mask.to(dtype=inputs.dtype)
            if token_mask.dtype != inputs.dtype:
                token_mask = token_mask.to(dtype=inputs.dtype)

            ort_inputs = numpy_inputs(
                input_names,
                (inputs, pooled_text_features, token_features, token_mask, text_mask),
            )
            input_pack_elapsed_ms = (time.perf_counter() - input_pack_start) * 1000.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            onnx_start = time.perf_counter()
            outputs = session.run(None, ort_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            onnx_elapsed_ms = (time.perf_counter() - onnx_start) * 1000.0

            post_start = time.perf_counter()
            output_tensor_start = time.perf_counter()
            cls_scores, bbox_preds = output_tensors(outputs, num_levels, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output_tensor_elapsed_ms = (time.perf_counter() - output_tensor_start) * 1000.0

            predict_start = time.perf_counter()
            model.bbox_head.num_classes = int(pooled_text_features.shape[1])
            results = model.bbox_head.predict_by_feat(
                cls_scores,
                bbox_preds,
                batch_img_metas=sample_metas(data_samples),
                rescale=True,
                with_nms=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            predict_elapsed_ms = (time.perf_counter() - predict_start) * 1000.0
            post_elapsed_ms = (time.perf_counter() - post_start) * 1000.0
            e2e_elapsed_ms = (time.perf_counter() - e2e_start) * 1000.0

            metric_start = time.perf_counter()
            pred_bbox, score = top_prediction(results[0])
            if score == 0.0 and pred_bbox == [0.0, 0.0, 0.0, 0.0]:
                n_empty += 1
            iou = compute_iou_np(pred_bbox, gt_bbox)
            is_correct = iou >= args.iou_threshold
            n_correct += int(is_correct)
            details.append({
                "idx": idx,
                "ann_id": ann.get("ann_id"),
                "image_path": ann.get("image_path"),
                "text": annotation_text(ann),
                "gt_bbox": [float(x) for x in gt_bbox],
                "pred_bbox": pred_bbox,
                "score": round(score, 8),
                "iou": round(float(iou), 8),
                "is_correct": bool(is_correct),
            })
            metric_elapsed_ms = (time.perf_counter() - metric_start) * 1000.0

            if idx >= args.warmup:
                preprocess_ms.append(preprocess_elapsed_ms)
                text_cache_ms.append(text_cache_elapsed_ms)
                input_pack_ms.append(input_pack_elapsed_ms)
                onnx_ms.append(onnx_elapsed_ms)
                output_tensor_ms.append(output_tensor_elapsed_ms)
                predict_by_feat_ms.append(predict_elapsed_ms)
                postprocess_ms.append(post_elapsed_ms)
                metric_ms.append(metric_elapsed_ms)
                e2e_ms.append(e2e_elapsed_ms)

    n_total = len(details)
    acc = 0.0 if n_total == 0 else n_correct / n_total * 100.0
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "onnx": args.onnx,
        "provider": args.provider,
        "ann_file": args.ann_file,
        "n_total": n_total,
        "n_correct": n_correct,
        "acc_at_0_5": round(acc, 4),
        "iou_threshold": args.iou_threshold,
        "empty_predictions": n_empty,
        "mean_iou": round(float(np.mean([row["iou"] for row in details])) if details else 0.0, 6),
        "warmup_excluded_from_latency": args.warmup,
        "latency": {
            "preprocess": latency_stats(preprocess_ms),
            "text_cache_lookup": latency_stats(text_cache_ms),
            "input_pack_numpy": latency_stats(input_pack_ms),
            "onnx_runtime": latency_stats(onnx_ms),
            "output_tensor_to_cuda": latency_stats(output_tensor_ms),
            "predict_by_feat": latency_stats(predict_by_feat_ms),
            "bbox_postprocess": latency_stats(postprocess_ms),
            "metric_acc_at_0_5": latency_stats(metric_ms),
            "e2e_preprocess_onnx_postprocess": latency_stats(e2e_ms),
        },
        "total_wall_sec": round(time.perf_counter() - total_start, 4),
    }
    mkdir_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    if args.details_output:
        mkdir_parent(args.details_output)
        with open(args.details_output, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "n_total": summary["n_total"],
        "n_correct": summary["n_correct"],
        "acc_at_0_5": summary["acc_at_0_5"],
        "e2e_mean_ms": summary["latency"]["e2e_preprocess_onnx_postprocess"]["mean_ms"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
