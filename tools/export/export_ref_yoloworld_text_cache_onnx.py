"""Export Ref-YOLO-World with query-specific cached text tensors.

This is the first ONNX path for ADR 0001. Cache lookup stays outside the
model; the exported graph receives image tensors and already-materialized text
side tensors for the current query.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from io import BytesIO
from typing import Sequence

import onnx
from onnx import TensorProto, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.apis import init_detector
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from torch import Tensor

warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings(action="ignore", category=torch.jit.ScriptWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=ResourceWarning)


class RefYOLOWorldTextCacheONNXWrapper(nn.Module):
    """Tensor-only wrapper for the text-cache inference structure."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self,
                images: Tensor,
                pooled_text_features: Tensor,
                token_features: Tensor,
                token_attention_mask: Tensor,
                text_mask: Tensor) -> tuple[Tensor, ...]:
        img_feats = self.base_model.backbone.forward_image(images)

        per_token_feats = token_features[:, 0]
        per_token_mask = token_attention_mask[:, 0]
        if self.base_model.with_neck:
            if self.base_model.mm_neck:
                neck_guide = pooled_text_features
                neck_guide_mask = None
                if self.base_model.neck_guide_positive_only:
                    neck_guide = per_token_feats
                    neck_guide_mask = per_token_mask
                img_feats = self.base_model.neck(img_feats, neck_guide,
                                                 neck_guide_mask)
            else:
                img_feats = self.base_model.neck(img_feats)

        self.base_model.bbox_head.num_classes = pooled_text_features.shape[1]
        cls_scores, bbox_preds = self.base_model.bbox_head.forward(
            img_feats,
            pooled_text_features,
            text_mask,
            per_token_feats=per_token_feats,
            per_token_mask=per_token_mask)

        if torch.onnx.is_in_onnx_export():
            cls_scores = [cls_score.float() for cls_score in cls_scores]
            bbox_preds = [bbox_pred.float() for bbox_pred in bbox_preds]

        outputs: list[Tensor] = []
        outputs.extend(cls_scores)
        outputs.extend(bbox_preds)
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Ref-YOLO-World ONNX with cached text tensors as inputs.")
    parser.add_argument("config", help="MMEngine config file")
    parser.add_argument("checkpoint", help="Model checkpoint file")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--summary-output", default=None,
                        help="Optional JSON summary path for experiment contract checks")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-text-slots", type=int, default=1)
    parser.add_argument("--max-token-len", type=int, default=77)
    parser.add_argument("--embed-dims", type=int, default=None)
    parser.add_argument("--text-cache", default=None,
                        help="Optional text cache .pt file used to seed dummy text tensors")
    parser.add_argument("--cache-indices", nargs="+", type=int, default=None,
                        help="Cache row indices used when --text-cache is provided")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true",
                        help="Export dynamic batch and token length axes")
    parser.add_argument("--check-onnx-runtime", action="store_true",
                        help="Load the exported model with ONNX Runtime and run a dummy forward")
    parser.add_argument("--tflite-strip-cls-score-where", action="store_true",
                        help=("Replace final cls_score Where guards with Identity outputs. "
                              "This avoids an onnx2tf shape expansion bug and preserves "
                              "finite-input scores for TFLite parity checks."))
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def _infer_embed_dims(model: nn.Module, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    head_module = getattr(getattr(model, "bbox_head", None), "head_module", None)
    embed_dims = getattr(head_module, "embed_dims", None)
    if embed_dims is None:
        raise ValueError("Cannot infer embed dims; pass --embed-dims explicitly.")
    return int(embed_dims)


def _cache_payload_tensors(payload: dict, indices: Sequence[int]) -> tuple[Tensor, Tensor, Tensor]:
    pooled = payload.get("pooled_text_features", payload.get("text_features"))
    tokens = payload.get("token_features", payload.get("per_token_features"))
    mask = payload.get("attention_mask", payload.get("token_attention_mask"))
    if pooled is None or tokens is None or mask is None:
        raise KeyError("Text cache must contain pooled_text_features, token_features, and attention_mask.")
    index = torch.as_tensor(indices, dtype=torch.long)
    return pooled.index_select(0, index), tokens.index_select(0, index), mask.index_select(0, index)


def build_dummy_inputs(args: argparse.Namespace,
                       model: nn.Module) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = torch.device(args.device)
    dtype = next(model.parameters()).dtype
    height, width = args.img_size
    images = torch.randn(args.batch_size, 3, height, width, device=device, dtype=dtype)

    total_slots = args.batch_size * args.num_text_slots
    if args.text_cache:
        payload = torch.load(args.text_cache, map_location="cpu")
        indices = args.cache_indices or list(range(args.num_text_slots))
        if len(indices) != args.num_text_slots:
            raise ValueError("--cache-indices length must match --num-text-slots.")
        repeated_indices = list(indices) * args.batch_size
        pooled, tokens, token_mask = _cache_payload_tensors(payload, repeated_indices)
        pooled = pooled.view(args.batch_size, args.num_text_slots, -1)
        tokens = tokens.view(args.batch_size, args.num_text_slots, tokens.shape[1], -1)
        token_mask = token_mask.view(args.batch_size, args.num_text_slots, -1)
    else:
        embed_dims = _infer_embed_dims(model, args.embed_dims)
        pooled = torch.randn(total_slots, embed_dims)
        pooled = F.normalize(pooled, dim=-1).view(args.batch_size, args.num_text_slots, embed_dims)
        tokens = torch.randn(args.batch_size, args.num_text_slots, args.max_token_len, embed_dims)
        token_mask = torch.ones(args.batch_size, args.num_text_slots, args.max_token_len)

    pooled = pooled.to(device=device, dtype=dtype)
    tokens = tokens.to(device=device, dtype=dtype)
    token_mask = token_mask.to(device=device, dtype=dtype)
    text_mask = torch.ones(args.batch_size, args.num_text_slots, device=device, dtype=dtype)
    return images, pooled, tokens, token_mask, text_mask


def output_names(num_levels: int) -> list[str]:
    cls_names = [f"cls_score_s{i}" for i in range(num_levels)]
    bbox_names = [f"bbox_pred_s{i}" for i in range(num_levels)]
    return cls_names + bbox_names


def dynamic_axes(names: Sequence[str], output_names_: Sequence[str]) -> dict[str, dict[int, str]]:
    axes = {
        "images": {0: "batch", 2: "height", 3: "width"},
        "pooled_text_features": {0: "batch", 1: "num_text_slots"},
        "token_features": {0: "batch", 1: "num_text_slots", 2: "token_len"},
        "token_attention_mask": {0: "batch", 1: "num_text_slots", 2: "token_len"},
        "text_mask": {0: "batch", 1: "num_text_slots"},
    }
    for name in output_names_:
        axes[name] = {0: "batch"}
    return {name: axes[name] for name in names + list(output_names_) if name in axes}


def normalize_onnx_float_types(onnx_model: onnx.ModelProto) -> dict[str, int]:
    stats = {
        "onnx_float_normalized_initializers": 0,
        "onnx_float_normalized_constants": 0,
        "onnx_float_normalized_casts": 0,
    }

    for idx, initializer in enumerate(onnx_model.graph.initializer):
        if initializer.data_type != TensorProto.DOUBLE:
            continue
        value = numpy_helper.to_array(initializer).astype("float32")
        onnx_model.graph.initializer[idx].CopyFrom(
            numpy_helper.from_array(value, initializer.name))
        stats["onnx_float_normalized_initializers"] += 1

    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type != onnx.AttributeProto.TENSOR:
                    continue
                if attr.t.data_type != TensorProto.DOUBLE:
                    continue
                value = numpy_helper.to_array(attr.t).astype("float32")
                attr.t.CopyFrom(numpy_helper.from_array(value, attr.t.name))
                stats["onnx_float_normalized_constants"] += 1
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    stats["onnx_float_normalized_casts"] += 1

    return stats


def strip_cls_score_output_where_guards(onnx_model: onnx.ModelProto,
                                        output_names_: Sequence[str]) -> int:
    """Bypass final cls_score Where guards for TFLite conversion.

    onnx2tf 1.28.x expands the final Ref-YOLO-World cls_score Where outputs
    from [B, C, H, W] to [B, H, W, H]. The Where nodes only guard finite
    values at the public outputs. For the finite model outputs validated by
    ONNX Runtime, their data input is identical to the Where output, while the
    direct Identity output keeps the static [B, C, H, W] shape in TFLite.
    """
    cls_output_names = {
        name for name in output_names_ if name.startswith("cls_score_")
    }
    graph_output_names = {output.name for output in onnx_model.graph.output}
    producer_by_output = {
        output_name: node
        for node in onnx_model.graph.node
        for output_name in node.output
    }
    stripped = 0
    for output_name in sorted(cls_output_names & graph_output_names):
        producer = producer_by_output.get(output_name)
        if producer is None or producer.op_type != "Where":
            continue
        replacement = producer.input[-1]
        producer.output[:] = [
            f"{value}_where_guard_unused" if value == output_name else value
            for value in producer.output
        ]
        identity_name = f"{output_name}_identity_output"
        onnx_model.graph.node.append(
            onnx.helper.make_node(
                "Identity", [replacement], [output_name], name=identity_name))
        stripped += 1
    return stripped

def check_onnx_runtime(onnx_path: str,
                       input_names: Sequence[str],
                       dummy_inputs: Sequence[Tensor]) -> dict:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, dummy_inputs)
    }
    outputs = session.run(None, ort_inputs)
    return {
        "onnxruntime_provider": "CPUExecutionProvider",
        "onnxruntime_load_pass": True,
        "onnxruntime_forward_pass": True,
        "onnxruntime_output_count": len(outputs),
        "onnxruntime_outputs_finite": all(np.isfinite(output).all() for output in outputs),
    }


def main() -> None:
    args = parse_args()
    mkdir_or_exist(os.path.dirname(args.output) or ".")

    base_model = init_detector(args.config, args.checkpoint, device=args.device)
    base_model.eval()
    wrapper = RefYOLOWorldTextCacheONNXWrapper(base_model).eval()
    dummy_inputs = build_dummy_inputs(args, base_model)

    input_names = [
        "images",
        "pooled_text_features",
        "token_features",
        "token_attention_mask",
        "text_mask",
    ]
    names = output_names(len(base_model.bbox_head.head_module.featmap_strides))

    with torch.no_grad():
        wrapper(*dummy_inputs)
        with BytesIO() as buffer:
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                buffer,
                input_names=input_names,
                output_names=names,
                dynamic_axes=dynamic_axes(input_names, names) if args.dynamic else None,
                opset_version=args.opset)
            buffer.seek(0)
            onnx_model = onnx.load(buffer)
            float_normalize_summary = normalize_onnx_float_types(onnx_model)
            tflite_stripped_cls_score_where = 0
            if args.tflite_strip_cls_score_where:
                tflite_stripped_cls_score_where = strip_cls_score_output_where_guards(
                    onnx_model, names)
            onnx.checker.check_model(onnx_model)

    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "onnxsim check failed"
        except Exception as exc:
            print_log(f"Simplify failure: {exc}")

    onnx.save(onnx_model, args.output)
    runtime_summary = {
        "onnxruntime_check_enabled": bool(args.check_onnx_runtime),
        "onnxruntime_load_pass": None,
        "onnxruntime_forward_pass": None,
        "onnxruntime_outputs_finite": None,
        "onnxruntime_error": None,
    }
    runtime_error: Exception | None = None
    if args.check_onnx_runtime:
        try:
            runtime_summary.update(
                check_onnx_runtime(args.output, input_names, dummy_inputs))
        except Exception as exc:
            runtime_summary.update({
                "onnxruntime_load_pass": False,
                "onnxruntime_forward_pass": False,
                "onnxruntime_outputs_finite": False,
                "onnxruntime_error": str(exc),
            })
            runtime_error = exc

    if args.summary_output:
        mkdir_or_exist(os.path.dirname(args.summary_output) or ".")
        summary = {
            "onnx_path": args.output,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "input_names": input_names,
            "output_names": names,
            "batch_size": args.batch_size,
            "num_text_slots": args.num_text_slots,
            "max_token_len": int(dummy_inputs[2].shape[2]),
            "opset": args.opset,
            "dynamic": bool(args.dynamic),
            "uses_external_text_tensors": True,
            "cache_lookup_inside_onnx": False,
            "tflite_strip_cls_score_where": bool(args.tflite_strip_cls_score_where),
            "tflite_stripped_cls_score_where": tflite_stripped_cls_score_where,
            **float_normalize_summary,
            **runtime_summary,
        }
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    if runtime_error is not None:
        raise runtime_error
    print_log(f"ONNX export success: {args.output}")


if __name__ == "__main__":
    main()
