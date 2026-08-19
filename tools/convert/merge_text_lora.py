#!/usr/bin/env python3
"""Merge the CLIP text LoRA model weights into base linear layers."""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmyolo.registry import MODELS


TEXT_MODEL_PREFIXES = ("backbone.text_model.", "module.backbone.text_model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge text LoRA weights into a Ref-YOLO-World text-LoRA checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-queries", type=int, default=100)
    return parser.parse_args()


def _iter_queries(items: list[dict[str, Any]]):
    seen: set[str] = set()
    for item in items:
        values = item.get("captions") or []
        if not values:
            for sent in item.get("sentences") or []:
                if isinstance(sent, dict):
                    values.append(sent.get("sent", sent.get("raw", "")))
                else:
                    values.append(str(sent))
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                yield [[text], [" "]]


def _load_config(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)
    return cfg


def _extract_text_state_dict(state_dict: dict[str, Any]) -> OrderedDict[str, Any]:
    text_state_dict: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        for prefix in TEXT_MODEL_PREFIXES:
            if key.startswith(prefix):
                text_state_dict[key[len(prefix):]] = value
                break
    return text_state_dict


def _load_checkpoint(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_text_model(
    model_cfg: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
):
    text_model = MODELS.build(model_cfg)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    text_state_dict = _extract_text_state_dict(state_dict)
    missing, unexpected = text_model.load_state_dict(text_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected text_model keys: {unexpected}")
    critical_missing = [key for key in missing if "lora_" in key]
    if critical_missing:
        raise RuntimeError(f"Missing LoRA weights in checkpoint load: {critical_missing}")
    text_model.to(device)
    text_model.eval()
    return text_model


def _clone_base_text_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    base_cfg = copy.deepcopy(model_cfg)
    base_cfg["type"] = "HuggingCLIPLanguageBackbonePerToken"
    for key in (
        "lora_target_layers",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    ):
        base_cfg.pop(key, None)
    return base_cfg


def _rename_linear_key(key: str) -> str:
    return key.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")


def _merge_state_dict(
    state_dict: dict[str, Any],
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    merged: OrderedDict[str, Any] = OrderedDict()
    merge_pairs: list[dict[str, Any]] = []

    for key, value in state_dict.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if ".linear.weight" in key:
            prefix = key[: -len(".linear.weight")]
            lora_a_key = f"{prefix}.lora_A"
            lora_b_key = f"{prefix}.lora_B"
            weight = value.detach().clone()
            if lora_a_key in state_dict and lora_b_key in state_dict:
                lora_a = state_dict[lora_a_key].detach().float()
                lora_b = state_dict[lora_b_key].detach().float()
                rank = int(lora_a.shape[0])
                scaling = 32.0 / float(rank)
                delta = (scaling * (lora_b @ lora_a)).to(dtype=weight.dtype)
                weight = weight + delta
                merge_pairs.append(
                    {
                        "prefix": prefix,
                        "rank": rank,
                        "scaling": scaling,
                        "delta_abs_max": float(delta.abs().max().item()),
                    })
            merged[_rename_linear_key(key)] = weight
            continue
        if ".linear.bias" in key:
            merged[_rename_linear_key(key)] = value
            continue
        merged[key] = value

    return merged, {
        "n_merged_pairs": len(merge_pairs),
        "pairs": merge_pairs,
    }


def _max_abs_diff(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or rhs is None:
        return float("inf")
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if tuple(lhs.shape) != tuple(rhs.shape):
        return float("inf")
    return float((lhs - rhs).abs().max().item())


def _evaluate_text_model(
    text_model,
    queries: list[list[list[str]]],
) -> dict[str, Any]:
    with torch.no_grad():
        for texts in queries:
            pooled, mask = text_model(texts)
            token = getattr(text_model, "_last_per_token", None)
            token_mask = getattr(text_model, "_last_per_token_attention_mask", None)
            yield {
                "pooled": pooled.detach().cpu(),
                "mask": None if mask is None else mask.detach().cpu(),
                "token": None if token is None else token.detach().cpu(),
                "token_mask": None if token_mask is None else token_mask.detach().cpu(),
            }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = _load_config(args.config)

    checkpoint = _load_checkpoint(args.checkpoint)
    state_dict = checkpoint.get("state_dict", checkpoint)
    merged_state_dict, merge_metadata = _merge_state_dict(state_dict)

    lora_text_model = _build_text_model(
        cfg.model.backbone.text_model,
        args.checkpoint,
        device,
    )
    base_text_cfg = _clone_base_text_model_cfg(cfg.model.backbone.text_model)
    merged_text_model = MODELS.build(base_text_cfg)

    merged_text_state_dict = _extract_text_state_dict(merged_state_dict)
    missing, unexpected = merged_text_model.load_state_dict(
        merged_text_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected merged text_model keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing merged text_model keys: {missing}")
    merged_text_model.to(device)
    merged_text_model.eval()

    with open(args.ann_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    queries = []
    for texts in _iter_queries(items):
        queries.append(texts)
        if len(queries) >= args.verify_queries:
            break
    if not queries:
        raise RuntimeError(f"No non-empty queries found in {args.ann_file}")

    pooled_max = 0.0
    token_max = 0.0
    mask_max = 0.0
    token_mask_max = 0.0
    for before, after in zip(
            _evaluate_text_model(lora_text_model, queries),
            _evaluate_text_model(merged_text_model, queries)):
        pooled_max = max(pooled_max, _max_abs_diff(before["pooled"], after["pooled"]))
        token_max = max(token_max, _max_abs_diff(before["token"], after["token"]))
        mask_max = max(mask_max, _max_abs_diff(before["mask"], after["mask"]))
        token_mask_max = max(
            token_mask_max, _max_abs_diff(before["token_mask"], after["token_mask"]))

    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    merged_checkpoint = copy.deepcopy(checkpoint)
    if isinstance(merged_checkpoint, dict) and "state_dict" in merged_checkpoint:
        merged_checkpoint["state_dict"] = merged_state_dict
    else:
        merged_checkpoint = merged_state_dict
    torch.save(merged_checkpoint, output_checkpoint)

    output = {
        "merged_checkpoint": str(output_checkpoint),
        "n_queries_verified": len(queries),
        "merge_text_max_abs_diff": round(
            max(pooled_max, token_max, mask_max, token_mask_max), 8),
        "pooled_text_max_abs_diff": round(pooled_max, 8),
        "per_token_max_abs_diff": round(token_max, 8),
        "text_mask_max_abs_diff": round(mask_max, 8),
        "per_token_mask_max_abs_diff": round(token_mask_max, 8),
        "merge_metadata": merge_metadata,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
