"""Build ADR 0001 text feature cache for Ref-YOLO-World.

This script precomputes CLIP pooled features, per-token features, and attention
masks for referring expressions. The saved file can be consumed by
``RefYOLOWorldDetector(text_feature_cache=...)``.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from mmdet.apis import init_detector
from mmyolo.utils import register_all_modules
from mmengine.config import Config
from transformers import AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP


TEXT_MODEL_PREFIXES = ("backbone.text_model.", "module.backbone.text_model.")
DEFAULT_EXTRA_TEXTS = (" ",)


def _pad_token_chunk(
    token_chunk: torch.Tensor,
    mask_chunk: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_length = int(token_chunk.shape[1])
    if current_length == target_length:
        return token_chunk, mask_chunk
    if current_length > target_length:
        return (
            token_chunk[:, :target_length],
            mask_chunk[:, :target_length],
        )

    pad_tokens = token_chunk.new_zeros(
        (token_chunk.shape[0], target_length - current_length, token_chunk.shape[2]))
    pad_mask = mask_chunk.new_zeros((mask_chunk.shape[0], target_length - current_length))
    return (
        torch.cat((token_chunk, pad_tokens), dim=1),
        torch.cat((mask_chunk, pad_mask), dim=1),
    )


def _resolve_save_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported --save-dtype: {name}")
    return mapping[key]


def _tensor_manifest(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
    }


def _iter_expressions(items: list[dict[str, Any]]):
    for item_index, item in enumerate(items):
        ann_id = item.get("ann_id", item.get("ref_id", item_index))
        sentences = item.get("sentences", [])
        captions = item.get("captions", [])
        values = []
        if sentences:
            for sent_index, sent_item in enumerate(sentences):
                if isinstance(sent_item, dict):
                    text = sent_item.get("sent", sent_item.get("raw", ""))
                    expr_id = sent_item.get("sent_id", sent_item.get("sentId"))
                else:
                    text = str(sent_item)
                    expr_id = None
                values.append((sent_index, expr_id, text))
        elif captions:
            for sent_index, text in enumerate(captions):
                values.append((sent_index, None, str(text)))

        for sent_index, expr_id, text in values:
            text = str(text).strip()
            if not text:
                continue
            stable_id = str(expr_id) if expr_id is not None else f"{ann_id}:{sent_index}"
            yield stable_id, str(ann_id), sent_index, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ref-YOLO-World text feature cache.")
    parser.add_argument("--ann-file", required=True, help="RefCOCO annotation JSON")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-config", help="MMEngine config whose model.backbone.text_model should be cached")
    parser.add_argument("--checkpoint", help="Checkpoint used to load the configured text backbone")
    parser.add_argument(
        "--model-state",
        help="Alias for --checkpoint that avoids experiment-contract checkpoint parsing.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--mask-special-tokens", action="store_true")
    parser.add_argument("--save-dtype", default="float16", choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--manifest-output", help="Output JSON manifest path. Defaults to <output>.manifest.json")
    return parser.parse_args()


def _load_configured_text_model(args: argparse.Namespace, device: torch.device):
    checkpoint_path = args.checkpoint or args.model_state
    if not args.model_config or not checkpoint_path:
        raise ValueError("--model-config and --checkpoint must be specified together")

    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmyolo.registry import MODELS

    cfg = Config.fromfile(args.model_config)
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)

    text_model = MODELS.build(cfg.model.backbone.text_model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    text_state_dict = {}
    for key, value in state_dict.items():
        for prefix in TEXT_MODEL_PREFIXES:
            if key.startswith(prefix):
                text_state_dict[key[len(prefix):]] = value
                break

    if not text_state_dict:
        raise RuntimeError(f"No backbone.text_model weights found in {checkpoint_path}")

    missing, unexpected = text_model.load_state_dict(text_state_dict, strict=False)
    critical_missing = [key for key in missing if "lora_" in key]
    if critical_missing:
        raise RuntimeError(f"Missing LoRA weights in checkpoint load: {critical_missing}")
    if unexpected:
        print(f"ignored unexpected text_model keys: {len(unexpected)}")

    text_model.to(device)
    text_model.eval()
    return text_model.tokenizer, text_model.model, f"mmengine:{args.model_config}", checkpoint_path


def _load_detector_without_cache(
    args: argparse.Namespace,
    device: torch.device,
):
    checkpoint_path = args.checkpoint or args.model_state
    if not args.model_config or not checkpoint_path:
        raise ValueError("--model-config and --checkpoint must be specified together")

    register_all_modules()
    cfg = Config.fromfile(args.model_config)
    model_cfg = cfg.model
    if "text_feature_cache" in model_cfg:
        model_cfg = model_cfg.copy()
        model_cfg.pop("text_feature_cache", None)
        cfg.model = model_cfg
    model = init_detector(cfg, checkpoint_path, device=device)
    model.eval()
    detector = model.module if hasattr(model, "module") else model
    return detector, checkpoint_path


def _load_hf_text_model(args: argparse.Namespace, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = CLIPTextConfig.from_pretrained(args.model_name, output_hidden_states=True)
    model = CLIPTP.from_pretrained(args.model_name, config=config).to(device)
    model.eval()
    return tokenizer, model, args.model_name, None


def main() -> None:
    args = parse_args()
    with open(args.ann_file, "r") as f:
        items = json.load(f)

    by_text: OrderedDict[str, dict[str, Any]] = OrderedDict()
    expr_id_to_text: dict[str, str] = {}
    ann_id_to_texts: dict[str, set[str]] = {}
    for expr_id, ann_id, sent_index, text in _iter_expressions(items):
        by_text.setdefault(text, {"expr_ids": [], "ann_ids": []})
        by_text[text]["expr_ids"].append(expr_id)
        by_text[text]["ann_ids"].append(ann_id)
        expr_id_to_text[expr_id] = text
        ann_id_to_texts.setdefault(ann_id, set()).add(text)

    # Many Ref-YOLO-World eval paths use a second blank negative slot:
    # texts = [[query], [" "]]. Cache it explicitly so strict lookup works.
    for text in DEFAULT_EXTRA_TEXTS:
        by_text.setdefault(text, {"expr_ids": [], "ann_ids": []})

    texts = list(by_text.keys())
    if not texts:
        raise RuntimeError(f"No expressions found in {args.ann_file}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    detector = None
    if args.model_config or args.checkpoint:
        detector, checkpoint_path = _load_detector_without_cache(args, device)
        tokenizer = None
        model = None
        text_encoder_name = f"mmengine:{args.model_config}"
    else:
        tokenizer, model, text_encoder_name, checkpoint_path = _load_hf_text_model(args, device)

    save_dtype = _resolve_save_dtype(args.save_dtype)
    pooled_chunks = []
    token_chunks = []
    mask_chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            if detector is not None:
                detector.reparameterize([[text] for text in batch_texts])
                pooled = detector._reparameterized_text_feats[:, 0]
                per_token = detector._reparameterized_per_token_feats
                attention_mask = detector._reparameterized_per_token_mask
            else:
                tokens = tokenizer(
                    text=batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length).to(device)
                outputs = model(**tokens)
                pooled = outputs.text_embeds
                pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    hidden = hidden_states[-1]
                else:
                    hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError("Text model output has no token hidden states")
                per_token = model.text_projection(hidden)
                per_token = per_token / per_token.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                attention_mask = tokens["attention_mask"].clone()
                if args.mask_special_tokens:
                    ids = tokens["input_ids"]
                    special = torch.zeros_like(ids, dtype=torch.bool)
                    if tokenizer.bos_token_id is not None:
                        special |= ids == tokenizer.bos_token_id
                    if tokenizer.eos_token_id is not None:
                        special |= ids == tokenizer.eos_token_id
                    attention_mask = attention_mask.masked_fill(special, 0)

            pooled_chunks.append(pooled.to(dtype=save_dtype).cpu())
            token_chunks.append(per_token.to(dtype=save_dtype).cpu())
            if attention_mask is None:
                raise RuntimeError("Configured text backbone did not expose per-token attention mask")
            mask_chunks.append(attention_mask.cpu())

    text_to_index = {text: index for index, text in enumerate(texts)}
    expr_id_to_index = {
        expr_id: text_to_index[text]
        for expr_id, text in expr_id_to_text.items()
    }
    for ann_id, texts_for_ann in ann_id_to_texts.items():
        if len(texts_for_ann) == 1:
            text = next(iter(texts_for_ann))
            expr_id_to_index.setdefault(ann_id, text_to_index[text])

    pooled_tensor = torch.cat(pooled_chunks, dim=0)
    max_token_length = max(int(chunk.shape[1]) for chunk in token_chunks)
    padded_chunks = [
        _pad_token_chunk(token_chunk, mask_chunk, max_token_length)
        for token_chunk, mask_chunk in zip(token_chunks, mask_chunks)
    ]
    token_tensor = torch.cat([token_chunk for token_chunk, _ in padded_chunks], dim=0)
    mask_tensor = torch.cat([mask_chunk for _, mask_chunk in padded_chunks], dim=0)
    payload = {
        "cache_version": 2,
        "pooled_text_features": pooled_tensor,
        "token_features": token_tensor,
        "attention_mask": mask_tensor,
        "texts": texts,
        "text_to_index": text_to_index,
        "expr_id_to_index": expr_id_to_index,
        "metadata": {
            "ann_file": args.ann_file,
            "model_name": args.model_name,
            "tokenizer_name": args.model_name,
            "model_config": args.model_config,
            "checkpoint": checkpoint_path,
            "text_encoder_name": text_encoder_name,
            "max_token_length": args.max_length,
            "num_texts": len(texts),
            "extra_texts": list(DEFAULT_EXTRA_TEXTS),
            "mask_special_tokens": bool(args.mask_special_tokens),
            "save_dtype": str(save_dtype),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    manifest = {
        "cache_version": payload["cache_version"],
        "output": str(output),
        "ann_file": args.ann_file,
        "model_name": args.model_name,
        "model_config": args.model_config,
        "checkpoint": checkpoint_path,
        "text_encoder_name": text_encoder_name,
        "tokenizer_name": args.model_name,
        "max_token_length": args.max_length,
        "mask_special_tokens": bool(args.mask_special_tokens),
        "num_texts": len(texts),
        "pooled_text_features": _tensor_manifest(pooled_tensor),
        "token_features": _tensor_manifest(token_tensor),
        "attention_mask": _tensor_manifest(mask_tensor),
        "total_tensor_bytes": int(
            pooled_tensor.numel() * pooled_tensor.element_size()
            + token_tensor.numel() * token_tensor.element_size()
            + mask_tensor.numel() * mask_tensor.element_size()),
    }
    manifest_output = Path(args.manifest_output) if args.manifest_output else output.with_suffix(".manifest.json")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"saved text feature cache: {output} ({len(texts)} unique texts)")
    print(f"saved text feature cache manifest: {manifest_output}")


if __name__ == "__main__":
    main()
