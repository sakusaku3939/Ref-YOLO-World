"""Text feature cache loader for Ref-YOLO-World inference."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor


def _string_map(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    return {str(value): index for index, value in enumerate(raw)}


def _first_existing(payload: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise KeyError(f"None of the required cache keys exist: {names}")


def _to_tensor(value: Any, name: str) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value).detach().cpu()
    except Exception as exc:
        raise TypeError(f"Cache entry {name!r} cannot be converted to Tensor") from exc


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"auto", "none", "preserve"}:
        return None
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported text feature cache dtype: {name}")
    return mapping[normalized]


def _tensor_bytes(tensor: Optional[Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


@dataclass
class TextFeatureCacheBatch:
    pooled_text_features: Tensor
    text_mask: Optional[Tensor]
    token_features: Tensor
    token_attention_mask: Optional[Tensor]


class TextFeatureCache:
    """Lookup pooled and per-token CLIP features by expression id or text.

    Expected tensor shapes are compatible with ADR 0001:
    pooled text features: [N, C], token features: [N, L, C], attention mask:
    [N, L]. Lookup returns batched tensors [B, K, C], [B, K, L, C], [B, K, L].
    The current Ref-Text Contrastive Head uses the input referring expression
    in slot 0; its token features are returned as [B, L, C].
    """

    def __init__(self,
                 path: str,
                 strict: bool = True,
                 allow_text_fallback: bool = True,
                 resident_device: str = 'cpu',
                 cache_dtype: Optional[str] = None,
                 pin_memory: bool = False,
                 profile_cache: bool = False,
                 debug_log_path: Optional[str] = None,
                 debug_first_batches: int = 8,
                 debug_log_interval: int = 500) -> None:
        self.path = path
        self.strict = strict
        self.allow_text_fallback = allow_text_fallback
        self.resident_device = resident_device
        self.cache_dtype = _resolve_dtype(cache_dtype)
        self.pin_memory = pin_memory
        self.profile_cache = profile_cache
        self.debug_log_path = debug_log_path
        self.debug_first_batches = debug_first_batches
        self.debug_log_interval = debug_log_interval
        self.lookup_calls = 0
        self.lookup_source_counts = {
            'meta_index': 0,
            'expr_id': 0,
            'text': 0,
            'miss': 0,
        }
        self._cache_device = torch.device('cpu')

        load_start = time.perf_counter()
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("Text feature cache must be a dict saved by torch.save().")

        self.pooled = _to_tensor(
            _first_existing(payload, (
                "pooled_text_features",
                "text_features",
                "txt_feats",
            )),
            "pooled_text_features",
        )
        self.tokens = _to_tensor(
            _first_existing(payload, (
                "token_features",
                "per_token_features",
                "per_token_feats",
                "all_token_features",
            )),
            "token_features",
        )
        mask = payload.get("attention_mask", payload.get("token_attention_mask", payload.get("per_token_mask")))
        self.attention_mask = None if mask is None else _to_tensor(mask, "attention_mask").long()

        if self.pooled.ndim != 2:
            raise ValueError(f"pooled text features must be [N, C], got {tuple(self.pooled.shape)}")
        if self.tokens.ndim != 3:
            raise ValueError(f"token features must be [N, L, C], got {tuple(self.tokens.shape)}")
        if self.tokens.shape[0] != self.pooled.shape[0] or self.tokens.shape[2] != self.pooled.shape[1]:
            raise ValueError(
                "pooled and token feature shapes are inconsistent: "
                f"pooled={tuple(self.pooled.shape)}, tokens={tuple(self.tokens.shape)}")
        if self.attention_mask is not None and tuple(self.attention_mask.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError(
                "attention mask must be [N, L], got "
                f"{tuple(self.attention_mask.shape)} for tokens {tuple(self.tokens.shape)}")

        if self.cache_dtype is not None:
            self.pooled = self.pooled.to(dtype=self.cache_dtype)
            self.tokens = self.tokens.to(dtype=self.cache_dtype)
        if self.pin_memory and torch.cuda.is_available():
            self.pooled = self.pooled.pin_memory()
            self.tokens = self.tokens.pin_memory()
            if self.attention_mask is not None:
                self.attention_mask = self.attention_mask.pin_memory()

        self.expr_id_to_index = _string_map(
            payload.get("expr_id_to_index", payload.get("expression_id_to_index")))
        if not self.expr_id_to_index:
            self.expr_id_to_index = _string_map(payload.get("expr_ids", payload.get("expression_ids")))
        self.text_to_index = _string_map(payload.get("text_to_index"))
        if not self.text_to_index:
            self.text_to_index = _string_map(payload.get("texts"))
        self.metadata = dict(payload.get("metadata", {}))
        self._write_debug({
            'event': 'cache_loaded',
            'path': self.path,
            'load_sec': round(time.perf_counter() - load_start, 6),
            'resident_device': self.resident_device,
            'cache_dtype': str(self.cache_dtype),
            'pooled_shape': list(self.pooled.shape),
            'tokens_shape': list(self.tokens.shape),
            'attention_mask_shape': None if self.attention_mask is None else list(self.attention_mask.shape),
            'bytes': self._cache_bytes(),
            'metadata': self.metadata,
        })

    def lookup(self,
               texts: Sequence[Sequence[str]],
               metainfos: Optional[Sequence[dict[str, Any]]] = None,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None) -> Optional[TextFeatureCacheBatch]:
        if not texts:
            return None
        n_slots = len(texts[0])
        if any(len(item) != n_slots for item in texts):
            raise ValueError("All batch items must have the same number of text slots.")

        self.lookup_calls += 1
        call_id = self.lookup_calls
        lookup_start = time.perf_counter()
        indices: list[list[int]] = []
        sources: list[list[str]] = []
        misses: list[str] = []
        examples = []
        for batch_index, slots in enumerate(texts):
            meta = metainfos[batch_index] if metainfos is not None else {}
            row = []
            source_row = []
            for slot_index, text in enumerate(slots):
                index, source = self._lookup_one(text, slot_index, meta)
                self.lookup_source_counts[source] += 1
                if index is None:
                    misses.append(f"batch={batch_index}, slot={slot_index}, text={text!r}")
                    index = -1
                row.append(index)
                source_row.append(source)
            if call_id <= self.debug_first_batches and batch_index < 4:
                examples.append({
                    'batch_index': batch_index,
                    'ann_id': meta.get('ann_id'),
                    'expr_ids': meta.get('expr_ids', meta.get('expression_ids')),
                    'indices': row[:4],
                    'sources': source_row[:4],
                    'texts': [str(text)[:120] for text in slots[:2]],
                })
            indices.append(row)
            sources.append(source_row)
        lookup_sec = time.perf_counter() - lookup_start

        if misses:
            if self.strict:
                shown = "; ".join(misses[:8])
                extra = "" if len(misses) <= 8 else f" ... (+{len(misses) - 8} more)"
                self._write_debug({
                    'event': 'cache_miss',
                    'call': call_id,
                    'miss_count': len(misses),
                    'shown': shown,
                })
                raise KeyError(f"Text feature cache miss in {self.path}: {shown}{extra}")
            return None

        materialize_start = time.perf_counter()
        self._ensure_resident(device)
        materialize_sec = time.perf_counter() - materialize_start

        gather_start = time.perf_counter()
        index_device = self.pooled.device
        index_tensor = torch.tensor(indices, dtype=torch.long, device=index_device)
        flat_index = index_tensor.reshape(-1)
        batch_size = len(indices)
        pooled = self.pooled.index_select(0, flat_index).view(batch_size, n_slots, -1)
        tokens = self.tokens.index_select(0, flat_index).view(batch_size, n_slots, self.tokens.shape[1], -1)
        token_mask = None
        if self.attention_mask is not None:
            token_mask = self.attention_mask.index_select(0, flat_index).view(batch_size, n_slots, -1)
            max_valid_tokens = int(token_mask.sum(dim=-1).max().item())
            max_valid_tokens = max(1, min(max_valid_tokens, tokens.shape[2]))
            tokens = tokens[:, :, :max_valid_tokens]
            token_mask = token_mask[:, :, :max_valid_tokens]
        gather_sec = time.perf_counter() - gather_start

        text_mask = self._make_text_mask(texts)
        transfer_start = time.perf_counter()
        if dtype is not None and pooled.dtype != dtype:
            pooled = pooled.to(dtype=dtype)
            tokens = tokens.to(dtype=dtype)
        if device is not None and pooled.device != device:
            pooled = pooled.to(device=device, non_blocking=True)
            tokens = tokens.to(device=device, non_blocking=True)
            if token_mask is not None:
                token_mask = token_mask.to(device=device, non_blocking=True)
        if text_mask is not None:
            if device is not None:
                text_mask = text_mask.to(device=device, non_blocking=True)
            if dtype is not None:
                text_mask = text_mask.to(dtype=dtype)
        transfer_sec = time.perf_counter() - transfer_start

        if self._should_debug(call_id):
            flat_sources = [source for row in sources for source in row]
            source_counts = {source: flat_sources.count(source) for source in sorted(set(flat_sources))}
            self._write_debug({
                'event': 'lookup',
                'call': call_id,
                'batch_size': batch_size,
                'n_slots': n_slots,
                'lookup_sec': round(lookup_sec, 6),
                'materialize_sec': round(materialize_sec, 6),
                'gather_sec': round(gather_sec, 6),
                'transfer_sec': round(transfer_sec, 6),
                'cache_device': str(self.pooled.device),
                'output_device': str(pooled.device),
                'output_dtype': str(pooled.dtype),
                'source_counts': source_counts,
                'cumulative_source_counts': dict(self.lookup_source_counts),
                'examples': examples,
            })

        return TextFeatureCacheBatch(
            pooled_text_features=pooled,
            text_mask=text_mask,
            token_features=tokens,
            token_attention_mask=token_mask,
        )

    def write_debug(self, event: dict[str, Any]) -> None:
        self._write_debug(event)

    def should_log(self, call_id: int) -> bool:
        return self._should_debug(call_id)

    def _lookup_one(self, text: str, slot_index: int, meta: dict[str, Any]) -> tuple[Optional[int], str]:
        direct = meta.get('text_cache_indices', meta.get('text_feature_cache_indices'))
        if isinstance(direct, (list, tuple)) and slot_index < len(direct):
            return int(direct[slot_index]), 'meta_index'
        if slot_index == 0:
            scalar = meta.get('text_cache_index', meta.get('text_feature_cache_index'))
            if scalar is not None:
                return int(scalar), 'meta_index'

        for key in self._explicit_expr_key_candidates(slot_index, meta):
            if key in self.expr_id_to_index:
                return self.expr_id_to_index[key], 'expr_id'

        if self.allow_text_fallback:
            index = self.text_to_index.get(str(text))
            if index is not None:
                return index, 'text'

        for key in self._ann_id_key_candidates(slot_index, meta):
            if key in self.expr_id_to_index:
                return self.expr_id_to_index[key], 'expr_id'
        return None, 'miss'

    def _explicit_expr_key_candidates(self, slot_index: int, meta: dict[str, Any]) -> Iterable[str]:
        explicit = meta.get("expr_ids", meta.get("expression_ids"))
        if isinstance(explicit, (list, tuple)) and slot_index < len(explicit):
            yield str(explicit[slot_index])
        elif explicit is not None and slot_index == 0:
            yield str(explicit)

    def _ann_id_key_candidates(self, slot_index: int, meta: dict[str, Any]) -> Iterable[str]:
        ann_id = meta.get("ann_id", meta.get("ref_id", meta.get("img_id")))
        if ann_id is not None:
            yield f"{ann_id}:{slot_index}"
            if slot_index == 0:
                yield str(ann_id)

    def _ensure_resident(self, runtime_device: Optional[torch.device]) -> None:
        target = self._target_device(runtime_device)
        if target.type == 'cpu':
            return
        if self.pooled.device == target and self.tokens.device == target:
            return
        start = time.perf_counter()
        self.pooled = self.pooled.to(device=target, non_blocking=True)
        self.tokens = self.tokens.to(device=target, non_blocking=True)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device=target, non_blocking=True)
        self._cache_device = target
        self._write_debug({
            'event': 'cache_materialized',
            'device': str(target),
            'sec': round(time.perf_counter() - start, 6),
            'bytes': self._cache_bytes(),
            'pooled_dtype': str(self.pooled.dtype),
            'tokens_dtype': str(self.tokens.dtype),
        })

    def _target_device(self, runtime_device: Optional[torch.device]) -> torch.device:
        resident = str(self.resident_device).lower()
        if resident in {'cpu', 'none'}:
            return torch.device('cpu')
        if resident in {'cuda', 'gpu'}:
            if runtime_device is not None and runtime_device.type == 'cuda':
                return runtime_device
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.resident_device)

    def _cache_bytes(self) -> int:
        return _tensor_bytes(self.pooled) + _tensor_bytes(self.tokens) + _tensor_bytes(self.attention_mask)

    def _should_debug(self, call_id: int) -> bool:
        if not self.debug_log_path:
            return False
        if call_id <= self.debug_first_batches:
            return True
        return self.debug_log_interval > 0 and call_id % self.debug_log_interval == 0

    def _write_debug(self, event: dict[str, Any]) -> None:
        if not self.debug_log_path:
            return
        payload = {
            'time': time.time(),
            **event,
        }
        directory = os.path.dirname(self.debug_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')

    @staticmethod
    def _make_text_mask(texts: Sequence[Sequence[str]]) -> Optional[Tensor]:
        mask = [[1.0 if str(text) != "" else 0.0 for text in slots] for slots in texts]
        if all(all(value == 1.0 for value in row) for row in mask):
            return None
        return torch.tensor(mask, dtype=torch.float32)
