"""Convert the LoRA-merged Ref-YOLO-World CLIP text encoder to Core ML.

The converted model takes CLIP token ids and an attention mask, and returns the
L2-normalized pooled and per-token text features that the Core ML detector from
``convert_ref_yoloworld_coreml.py`` expects. Merge the text LoRA first with
``tools/convert/merge_text_lora.py``. Before conversion, the script checks the
PyTorch outputs against a text-feature cache built from the same checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

MODEL_NAME = "openai/clip-vit-base-patch32"
CHECKPOINT_PREFIX = "backbone.text_model.model."
CACHE_ATOL = 2e-3


class CoreMLTextEncoder(torch.nn.Module):
    """CLIP text encoder that returns normalized pooled and per-token features."""

    def __init__(self, model: CLIPTextModelWithProjection) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_model(
            input_ids=input_ids.to(torch.int64),
            attention_mask=attention_mask.to(torch.int64),
            output_hidden_states=True)
        pooled = self.text_projection(outputs.pooler_output)
        # Per-token features use the last encoder hidden state before the final
        # LayerNorm, matching the text-feature cache.
        tokens = self.text_projection(outputs.hidden_states[-1])
        pooled = pooled / torch.linalg.vector_norm(
            pooled, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
        tokens = tokens / torch.linalg.vector_norm(
            tokens, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return pooled, tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Ref-YOLO-World CLIP text encoder to Core ML.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="LoRA-merged checkpoint from merge_text_lora.py")
    parser.add_argument("--cache", type=Path, required=True,
                        help="Text-feature cache built from the same checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .mlpackage path")
    parser.add_argument("--tokenizer-output", type=Path, required=True,
                        help="Directory for vocab.json and merges.txt")
    parser.add_argument("--sequence-length", type=int, default=77)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_model(checkpoint_path: Path) -> CLIPTextModelWithProjection:
    config = CLIPTextConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, config=config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    text_state = {
        key[len(CHECKPOINT_PREFIX):]: value
        for key, value in source.items() if key.startswith(CHECKPOINT_PREFIX)
    }
    missing, unexpected = model.load_state_dict(text_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Text checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()


def compare_with_cache(model: CoreMLTextEncoder, tokenizer: AutoTokenizer,
                       cache_path: Path, sequence_length: int) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    text = cache["texts"][0]
    encoded = tokenizer(text=[text], return_tensors="pt", padding="max_length",
                        truncation=True, max_length=sequence_length)
    with torch.no_grad():
        pooled, tokens = model(encoded["input_ids"], encoded["attention_mask"])
    expected_pooled = cache["pooled_text_features"][0].float().unsqueeze(0)
    expected_tokens = cache["token_features"][0, :sequence_length].float().unsqueeze(0)
    return {
        "prompt": text,
        "pooled_max_abs_error": float((pooled - expected_pooled).abs().max()),
        "token_max_abs_error": float((tokens - expected_tokens).abs().max()),
    }


def save_tokenizer(tokenizer: AutoTokenizer, output_dir: Path) -> None:
    staging = output_dir.with_name(f"{output_dir.name}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    tokenizer.save_pretrained(staging)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for file_name in ("vocab.json", "merges.txt"):
        shutil.copy2(staging / file_name, output_dir / file_name)
    shutil.rmtree(staging)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = CoreMLTextEncoder(load_text_model(args.checkpoint)).eval()

    comparison = compare_with_cache(encoder, tokenizer, args.cache, args.sequence_length)
    print(json.dumps(comparison, indent=2))
    if max(comparison["pooled_max_abs_error"],
           comparison["token_max_abs_error"]) > CACHE_ATOL:
        raise RuntimeError("Text encoder outputs do not match the text-feature cache")

    example_ids = torch.ones((1, args.sequence_length), dtype=torch.int32)
    example_ids[0, 0] = tokenizer.bos_token_id
    example_ids[0, 1] = tokenizer.eos_token_id
    example_mask = torch.zeros_like(example_ids)
    example_mask[0, :2] = 1
    traced = torch.jit.trace(encoder, (example_ids, example_mask), check_trace=False)

    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.sequence_length),
                          dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.sequence_length),
                          dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="pooled_text_features"),
            ct.TensorType(name="token_features"),
        ])
    model.short_description = "LoRA-merged CLIP ViT-B/32 text encoder for Ref-YOLO-World"
    checkpoint_sha256 = sha256(args.checkpoint)
    model.user_defined_metadata.update({
        "source_model": MODEL_NAME,
        "source_checkpoint_sha256": checkpoint_sha256,
        "sequence_length": str(args.sequence_length),
    })
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    save_tokenizer(tokenizer, args.tokenizer_output)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "output": str(args.output),
        "tokenizer_output": str(args.tokenizer_output),
        "sequence_length": args.sequence_length,
        "cache_comparison": comparison,
        "coremltools_version": ct.__version__,
        "torch_version": torch.__version__,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
