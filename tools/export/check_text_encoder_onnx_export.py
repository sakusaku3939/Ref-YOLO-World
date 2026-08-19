#!/usr/bin/env python3
"""Dry-run ONNX export for the merged CLIP text encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmyolo.registry import MODELS
from transformers import CLIPTextModelWithProjection as CLIPTP


TEXT_MODEL_PREFIXES = ("backbone.text_model.", "module.backbone.text_model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ONNX exportability of the merged CLIP text encoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--model-state",
        help="Alias for --checkpoint that avoids experiment-contract checkpoint parsing.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--onnx-output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="the person in a beige coat")
    return parser.parse_args()


def _load_config(path: str) -> Config:
    cfg = Config.fromfile(path)
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)
    return cfg


def _extract_text_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    text_state_dict: dict[str, Any] = {}
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


def _build_export_model(text_model: nn.Module, device: torch.device) -> nn.Module:
    model_name = getattr(text_model, "model_name", None)
    if model_name is None:
        model_name = getattr(getattr(text_model, "tokenizer", None), "name_or_path", None)
    if model_name is None:
        raise RuntimeError("Unable to resolve CLIP model name for ONNX export.")

    try:
        eager_model = CLIPTP.from_pretrained(
            model_name,
            config=text_model.model.config,
            attn_implementation="eager",
        )
        eager_model.load_state_dict(text_model.model.state_dict(), strict=True)
        eager_model.to(device)
        eager_model.eval()
        return eager_model
    except TypeError:
        fallback_model = text_model.model.to(device)
        fallback_model.eval()
        return fallback_model


class CLIPTextEncoderOnnxWrapper(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.text_embeds, outputs.last_hidden_state


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    device = torch.device(args.device)
    checkpoint_path = args.model_state or args.checkpoint
    if not checkpoint_path:
        raise ValueError("Either --checkpoint or --model-state must be specified.")

    text_model = MODELS.build(cfg.model.backbone.text_model)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    text_state_dict = _extract_text_state_dict(state_dict)
    missing, unexpected = text_model.load_state_dict(text_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected text_model keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing text_model keys: {missing}")

    hf_model = _build_export_model(text_model, device)
    export_model = CLIPTextEncoderOnnxWrapper(hf_model)
    export_model.eval()
    tokenizer = text_model.tokenizer
    tokens = tokenizer(
        text=[args.prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    exportable = False
    export_error = None
    try:
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                (tokens["input_ids"], tokens["attention_mask"]),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["text_embeds", "last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                },
                opset_version=17,
            )
        exportable = True
    except Exception as exc:  # pragma: no cover - runtime-only path
        export_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "onnx_exportable": exportable,
        "export_error": export_error,
        "onnx_output": str(onnx_path),
        "prompt": args.prompt,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
