"""Measure parameter counts of representative Ref-YOLO-World inference variants."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, os.getcwd())

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules


DEFAULT_CONFIG = (
    "configs/ref_yolo_world/ref_yolo_world_multidataset.py"
)
DEFAULT_TEXT_CACHE = (
    "results/text_cache/refcocog_val_text_features.pt"
)


@dataclass
class VariantResult:
    label: str
    total: int | None = None
    trainable: int | None = None
    buffers: int | None = None
    lora_params: int | None = None
    lora_trainable: int | None = None
    by_top: dict[str, dict[str, int]] | None = None
    folded_payload_numel: int | None = None
    status: str = "ok"
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure total/trainable parameter counts of Ref-YOLO-World."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--text-cache-path", default=DEFAULT_TEXT_CACHE)
    parser.add_argument("--folded-token-len", type=int, default=5)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--output", help="Optional output path.")
    return parser.parse_args()


def setup_registry() -> None:
    register_all_modules(init_default_scope=False)
    init_default_scope("mmyolo")
    import yolo_world.models  # noqa: F401


def load_config(path: str) -> Config:
    cfg = Config.fromfile(path)
    for module_name in cfg.get("custom_imports", {}).get("imports", []):
        importlib.import_module(module_name)
    return cfg


def count_folded_payload(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        params = getattr(module, "_folded_attn_params", None)
        if params is None:
            continue
        for value in vars(params).values():
            if isinstance(value, torch.Tensor):
                total += value.numel()
    return total


def count_model(model: torch.nn.Module) -> dict[str, Any]:
    by_top: dict[str, dict[str, int]] = {}
    for name in ("backbone", "neck", "bbox_head"):
        module = getattr(model, name, None)
        if module is None:
            continue
        by_top[name] = {
            "total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
        }
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "buffers": sum(b.numel() for b in model.buffers()),
        "lora_params": sum(
            p.numel()
            for name, p in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ),
        "lora_trainable": sum(
            p.numel()
            for name, p in model.named_parameters()
            if ("lora_A" in name or "lora_B" in name) and p.requires_grad
        ),
        "by_top": by_top,
        "folded_payload_numel": count_folded_payload(model),
    }


def make_no_lora_config(base_cfg: Config) -> Config:
    cfg = copy.deepcopy(base_cfg)
    text_model = cfg.model.backbone.text_model
    text_model.type = "HuggingCLIPLanguageBackbonePerToken"
    for key in (
        "lora_alpha",
        "lora_dropout",
        "lora_rank",
        "lora_target_layers",
        "lora_target_modules",
    ):
        text_model.pop(key, None)
    return cfg


def make_text_cache_config(base_cfg: Config, text_cache_path: str) -> Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.model.text_feature_cache = dict(
        path=text_cache_path,
        strict=True,
        allow_text_fallback=True,
        resident_device="cpu",
        cache_dtype="float16",
        pin_memory=False,
    )
    return cfg


def build_result(
    label: str,
    cfg: Config,
    post_build: Callable[[torch.nn.Module], None] | None = None,
) -> VariantResult:
    try:
        model = MODELS.build(cfg.model)
        if post_build is not None:
            post_build(model)
        return VariantResult(label=label, **count_model(model))
    except Exception as exc:  # pragma: no cover - keeps CLI useful across envs.
        return VariantResult(label=label, status="error", note=str(exc))


def merge_lora_weights(model: torch.nn.Module) -> None:
    text_model = getattr(getattr(model, "backbone", None), "text_model", None)
    merge = getattr(text_model, "merge_lora_weights", None)
    if merge is None:
        raise RuntimeError("backbone.text_model has no merge_lora_weights().")
    merge()


def set_folded_attention(model: torch.nn.Module, token_len: int) -> None:
    set_folded = getattr(getattr(model, "bbox_head", None), "set_folded_attn_params", None)
    if set_folded is None:
        raise RuntimeError("bbox_head has no set_folded_attn_params().")
    embed_dims = int(model.bbox_head.head_module.embed_dims)
    txt_feats = torch.randn(1, 1, embed_dims)
    per_token_feats = torch.randn(1, token_len, embed_dims)
    per_token_mask = torch.ones(1, token_len)
    set_folded(txt_feats, per_token_feats, per_token_mask)


def collect_results(args: argparse.Namespace) -> list[VariantResult]:
    setup_registry()
    base_cfg = load_config(args.config)
    results = [
        build_result("with LoRA", base_cfg),
        build_result("without LoRA", make_no_lora_config(base_cfg)),
        build_result("after LoRA merge", base_cfg, merge_lora_weights),
    ]
    if os.path.exists(args.text_cache_path):
        results.append(
            build_result(
                "with text cache",
                make_text_cache_config(base_cfg, args.text_cache_path),
            )
        )
    else:
        results.append(
            VariantResult(
                label="with text cache",
                status="skipped",
                note=f"text cache not found: {args.text_cache_path}",
            )
        )
    results.extend(
        [
            build_result("before attention reparameterization", base_cfg),
            build_result(
                "after attention reparameterization",
                base_cfg,
                lambda model: set_folded_attention(model, args.folded_token_len),
            ),
        ]
    )
    return results


def to_million(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000:.3f}M"


def results_to_markdown(results: list[VariantResult]) -> str:
    lines = [
        "| variant | total params | trainable params | LoRA params | status |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        status = result.status if not result.note else f"{result.status}: {result.note}"
        lines.append(
            "| "
            f"{result.label} | {to_million(result.total)} | "
            f"{to_million(result.trainable)} | {to_million(result.lora_params)} | "
            f"{status} |"
        )
    return "\n".join(lines) + "\n"


def results_to_json(results: list[VariantResult]) -> str:
    return json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    args = parse_args()
    results = collect_results(args)
    if args.format == "json":
        output = results_to_json(results)
    else:
        output = results_to_markdown(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
