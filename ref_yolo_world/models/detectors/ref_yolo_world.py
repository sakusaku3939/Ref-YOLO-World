# Ref-YOLO-World: YOLO-World Detector for referring expression.
# Passes per-token features of the input referring expression from the Backbone to the Head.
# They can also be passed to the Neck (enabled by neck_guide_positive_only).
# Works with HuggingCLIPLanguageBackbonePerToken (_last_per_token).

from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Union
import time

import torch
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

from ref_yolo_world.modules.text_feature_cache import TextFeatureCache


@MODELS.register_module()
class RefYOLOWorldDetector(YOLODetector):
    """YOLO-World Detector for referring expressions.

    After extract_feat(), takes per-token features from the backbone and passes them to
    bbox_head.
    When neck_guide_positive_only=True, the per-token features of the input referring
    expression are also passed to the Neck as guide (for RefYOLOWorldPAFPN + Token-Aware
    blocks).
    Despite the name, ``positive`` simply refers to the input referring expression;
    no negative expressions are involved.
    """

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 neck_guide_positive_only: bool = False,
                 text_feature_cache: Optional[dict] = None,
                 use_attention_reparameterization: Optional[bool] = None,
                 use_attention_reparameterization_cache: Optional[bool] = None,
                 attention_reparameterization_cache_max_entries: Optional[int] = None,
                 inference_autocast: Optional[dict] = None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.neck_guide_positive_only = neck_guide_positive_only
        self.text_feature_cache_cfg = text_feature_cache
        if attention_reparameterization_cache_max_entries is None:
            attention_reparameterization_cache_max_entries = 4096
        self.use_attention_reparameterization = bool(use_attention_reparameterization)
        self.use_attention_reparameterization_cache = bool(use_attention_reparameterization_cache)
        self.attention_reparameterization_cache_max_entries = int(
            attention_reparameterization_cache_max_entries)
        self.inference_autocast_cfg = dict(inference_autocast or {})
        self._inference_autocast_enabled = bool(
            self.inference_autocast_cfg.get('enabled', False))
        self._inference_autocast_dtype = self._resolve_autocast_dtype(
            self.inference_autocast_cfg.get('dtype', 'float16'))
        self._inference_autocast_scope = self._resolve_autocast_scope(
            self.inference_autocast_cfg)
        self._inference_fp32_modules = self._normalize_fp32_modules(
            self.inference_autocast_cfg.get('fp32_modules', []))
        rerun_cfg = dict(
            self.inference_autocast_cfg.get('low_margin_fp32_rerun') or {})
        self._low_margin_fp32_rerun_enabled = bool(
            rerun_cfg.get('enabled', False))
        self._low_margin_fp32_rerun_threshold = float(
            rerun_cfg.get('threshold', 0.0))
        self.text_feature_cache: Optional[TextFeatureCache] = None
        self._folded_attn_cache: dict[tuple, tuple] = {}
        self._active_folded_attn_key: Optional[tuple] = None
        self._cached_per_token_feats: Optional[Tensor] = None
        self._cached_per_token_mask: Optional[Tensor] = None
        self._reparameterized_text_feats: Optional[Tensor] = None
        self._reparameterized_text_masks: Optional[Tensor] = None
        self._reparameterized_per_token_feats: Optional[Tensor] = None
        self._reparameterized_per_token_mask: Optional[Tensor] = None
        self._cache_profile_calls = 0
        super().__init__(*args, **kwargs)
        if text_feature_cache is not None:
            cache_cfg = dict(text_feature_cache)
            cache_path = cache_cfg.pop('path')
            self.text_feature_cache = TextFeatureCache(cache_path, **cache_cfg)

    @property
    def use_attention_reparameterization(self) -> bool:
        return getattr(self, '_use_attention_reparameterization', False)

    @use_attention_reparameterization.setter
    def use_attention_reparameterization(self, value: bool) -> None:
        self._use_attention_reparameterization = bool(value)
        if (not self._use_attention_reparameterization
                and hasattr(self, 'bbox_head')):
            self._clear_folded_attn()

    @staticmethod
    def _split_text_outputs(
            text_outputs: Union[Tensor, Tuple[Tensor, Optional[Tensor]]]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(text_outputs, tuple):
            return text_outputs
        return text_outputs, None

    def _clear_folded_attn(self) -> None:
        if hasattr(self.bbox_head, 'set_folded_attn_params'):
            self.bbox_head.set_folded_attn_params(None, None, None)
        if self.with_neck:
            for module in self.neck.modules():
                if (hasattr(module, 'set_folded_attn')
                        and hasattr(module, '_folded_attn_params')):
                    module.set_folded_attn(None, None)
        self._active_folded_attn_key = None

    def _folded_attn_param_modules(self) -> List[torch.nn.Module]:
        return [
            module for module in self.modules()
            if hasattr(module, '_folded_attn_params')
        ]

    @staticmethod
    def _normalize_texts_key(texts: Optional[List[List[str]]]) -> Optional[tuple]:
        if texts is None:
            return None
        return tuple(tuple(str(text) for text in row) for row in texts)

    def _make_folded_attn_cache_key(
            self,
            texts: Optional[List[List[str]]],
            txt_feats: Tensor,
            per_token_feats: Tensor,
            per_token_mask: Optional[Tensor]) -> Optional[tuple]:
        texts_key = self._normalize_texts_key(texts)
        if texts_key is None:
            return None
        mask_shape = None if per_token_mask is None else tuple(per_token_mask.shape)
        mask_dtype = None if per_token_mask is None else str(per_token_mask.dtype)
        return (
            texts_key,
            str(txt_feats.dtype),
            tuple(txt_feats.shape),
            str(per_token_feats.dtype),
            tuple(per_token_feats.shape),
            mask_dtype,
            mask_shape,
        )

    def _capture_folded_attn_state(self) -> tuple:
        return tuple(
            getattr(module, '_folded_attn_params', None)
            for module in self._folded_attn_param_modules()
        )

    def _apply_folded_attn_state(self, state: tuple) -> None:
        modules = self._folded_attn_param_modules()
        if len(modules) != len(state):
            raise RuntimeError(
                'Folded attention cache is incompatible with current model: '
                f'{len(state)} cached modules vs {len(modules)} current modules')
        for module, params in zip(modules, state):
            module._folded_attn_params = params

    def _store_folded_attn_state(self, key: tuple) -> None:
        if not self.use_attention_reparameterization_cache:
            return
        if self.attention_reparameterization_cache_max_entries <= 0:
            return
        if key not in self._folded_attn_cache:
            if (len(self._folded_attn_cache) >=
                    self.attention_reparameterization_cache_max_entries):
                oldest_key = next(iter(self._folded_attn_cache))
                self._folded_attn_cache.pop(oldest_key, None)
            self._folded_attn_cache[key] = self._capture_folded_attn_state()

    def _refresh_folded_attn(
            self, txt_feats: Optional[Tensor],
            per_token_feats: Optional[Tensor],
            per_token_mask: Optional[Tensor],
            texts: Optional[List[List[str]]] = None) -> None:
        if not self.use_attention_reparameterization:
            self._clear_folded_attn()
            return
        if txt_feats is None or per_token_feats is None:
            self._clear_folded_attn()
            return
        cache_key = self._make_folded_attn_cache_key(
            texts, txt_feats, per_token_feats, per_token_mask)
        if cache_key is not None:
            if cache_key == self._active_folded_attn_key:
                return
            cached_state = self._folded_attn_cache.get(cache_key)
            if cached_state is not None:
                self._apply_folded_attn_state(cached_state)
                self._active_folded_attn_key = cache_key
                return
        if hasattr(self.bbox_head, 'set_folded_attn_params'):
            self.bbox_head.set_folded_attn_params(
                txt_feats, per_token_feats, per_token_mask)
        if self.with_neck:
            for module in self.neck.modules():
                if (hasattr(module, 'set_folded_attn')
                        and hasattr(module, '_folded_attn_params')):
                    module.set_folded_attn(per_token_feats, per_token_mask)
        if cache_key is not None:
            self._store_folded_attn_state(cache_key)
            self._active_folded_attn_key = cache_key
    @staticmethod
    def _resolve_autocast_dtype(dtype: Any) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype_map = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        resolved = dtype_map.get(str(dtype).lower())
        if resolved is None:
            raise ValueError(f'Unsupported inference autocast dtype: {dtype}')
        return resolved

    @staticmethod
    def _resolve_autocast_scope(cfg: dict[str, Any]) -> str:
        scope = cfg.get('scope')
        if scope is None:
            if not bool(cfg.get('reparameterize', True)):
                return 'forward'
            return 'both'
        scope = str(scope).lower()
        if scope not in {'both', 'forward', 'reparameterize'}:
            raise ValueError('Unsupported inference autocast scope: ' + scope)
        return scope

    @staticmethod
    def _normalize_fp32_modules(fp32_modules: Optional[List[str]]) -> tuple[str, ...]:
        alias_map = {
            'head_contrastive_scoring': 'head_contrastive',
            'head_contrastive': 'head_contrastive',
            'head_cross_attention': 'head_cross_attention',
            'neck_ref_t_csp': 'neck_ref_csp',
            'neck_ref_csp': 'neck_ref_csp',
            'spatial_pe': 'spatial_pe',
        }
        normalized = []
        for raw_name in fp32_modules or []:
            key = str(raw_name).strip().lower()
            resolved = alias_map.get(key)
            if resolved is None:
                raise ValueError(f'Unsupported fp32 module name: {raw_name}')
            if resolved not in normalized:
                normalized.append(resolved)
        return tuple(normalized)

    def set_inference_autocast(self,
                               enabled: bool,
                               dtype: Optional[Any] = None,
                               scope: Optional[str] = None,
                               fp32_modules: Optional[List[str]] = None) -> None:
        self._inference_autocast_enabled = bool(enabled)
        if dtype is not None:
            self._inference_autocast_dtype = self._resolve_autocast_dtype(dtype)
        if scope is not None:
            self._inference_autocast_scope = self._resolve_autocast_scope(
                {'scope': scope})
        if fp32_modules is not None:
            self._inference_fp32_modules = self._normalize_fp32_modules(
                fp32_modules)

    @staticmethod
    def _cast_tensor_tree_fp32(value):
        if isinstance(value, Tensor):
            if value.dtype.is_floating_point:
                return value.float()
            return value
        if isinstance(value, tuple):
            return tuple(
                RefYOLOWorldDetector._cast_tensor_tree_fp32(item)
                for item in value)
        if isinstance(value, list):
            return [
                RefYOLOWorldDetector._cast_tensor_tree_fp32(item)
                for item in value
            ]
        return value

    def _use_fp32_module(self, module_name: str) -> bool:
        return module_name in self._inference_fp32_modules and torch.cuda.is_available()

    def _sync_inference_fp32_modules(self) -> None:
        if hasattr(self.bbox_head, 'set_inference_fp32_modules'):
            self.bbox_head.set_inference_fp32_modules(
                self._inference_fp32_modules)

    def _inference_autocast_context(self, stage: str = 'forward'):
        if (not self._inference_autocast_enabled
                or not torch.cuda.is_available()):
            return nullcontext()
        if self._inference_autocast_scope != 'both' and stage != self._inference_autocast_scope:
            return nullcontext()
        return torch.amp.autocast(
            device_type='cuda',
            dtype=self._inference_autocast_dtype,
        )

    def _should_rerun_low_margin_fp32(self, results_list: list) -> bool:
        if (not self._low_margin_fp32_rerun_enabled
                or not self._inference_autocast_enabled
                or not torch.cuda.is_available()):
            return False
        threshold = self._low_margin_fp32_rerun_threshold
        if threshold <= 0:
            return False

        for result in results_list:
            scores = getattr(result, 'scores', None)
            if scores is None or scores.numel() < 2:
                continue
            top2 = torch.topk(scores.float(), k=2).values
            margin = float((top2[0] - top2[1]).detach().cpu())
            if margin <= threshold:
                return True
        return False

    def _predict_once(self,
                      batch_inputs: Tensor,
                      batch_data_samples: SampleList,
                      rescale: bool,
                      use_autocast: bool) -> tuple[list, float, float]:
        start = time.perf_counter()
        context = (self._inference_autocast_context()
                   if use_autocast else nullcontext())
        with context:
            img_feats, txt_feats, txt_masks = self.extract_feat(
                batch_inputs, batch_data_samples)
            extract_sec = time.perf_counter() - start
            per_token_feats, per_token_mask = self._get_per_token_feats()

            head_start = time.perf_counter()
            self.bbox_head.num_classes = txt_feats[0].shape[0]
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  txt_masks,
                                                  batch_data_samples,
                                                  rescale=rescale,
                                                  per_token_feats=per_token_feats,
                                                  per_token_mask=per_token_mask)
            head_sec = time.perf_counter() - head_start
        return results_list, extract_sec, head_sec

    def _get_per_token_feats(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get per-token features from the cache or the backbone text_model.

        When the text feature cache is enabled, tokens of the input referring expression
        taken from the cache are used first.
        Otherwise, reads _last_per_token / _last_per_token_attention_mask
        (HuggingCLIPLanguageBackbonePerToken), or _per_token_feats /
        _per_token_mask.
        """
        if self._cached_per_token_feats is not None:
            return self._cached_per_token_feats, self._cached_per_token_mask

        text_model = getattr(self.backbone, 'text_model', None)
        if text_model is None:
            return None, None
        feats = getattr(text_model, '_last_per_token', None)
        if feats is None:
            feats = getattr(text_model, '_per_token_feats', None)
        mask = getattr(text_model, '_last_per_token_attention_mask', None)
        if mask is None:
            mask = getattr(text_model, '_per_token_mask', None)
        return feats, mask

    @staticmethod
    def _metainfo_from_samples(batch_data_samples: SampleList) -> Optional[List[dict[str, Any]]]:
        if not isinstance(batch_data_samples, list):
            return None
        metainfos = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'metainfo'):
                metainfos.append(dict(data_sample.metainfo))
            else:
                metainfos.append({})
        return metainfos

    def _lookup_text_cache(self, texts: List[List[str]],
                           batch_data_samples: SampleList,
                           batch_inputs: Tensor):
        if self.text_feature_cache is None:
            return None
        return self.text_feature_cache.lookup(
            texts,
            metainfos=self._metainfo_from_samples(batch_data_samples),
            device=batch_inputs.device,
            dtype=batch_inputs.dtype)

    def _lookup_reparameterize_text_cache(self, texts: List[List[str]]):
        if self.text_feature_cache is None:
            return None
        param = next(self.parameters())
        return self.text_feature_cache.lookup(
            texts,
            metainfos=None,
            device=param.device,
            dtype=param.dtype)

    def _write_cache_debug(self, event: dict[str, Any]) -> None:
        if self.text_feature_cache is not None:
            self.text_feature_cache.write_debug(event)

    @staticmethod
    def _texts_match(current: Optional[List[List[str]]],
                     candidate: Optional[List[List[str]]]) -> bool:
        if current is None or candidate is None:
            return False
        return current == candidate

    def _resolve_runtime_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        try:
            param = next(self.parameters())
            return param.device, param.dtype
        except StopIteration:
            return torch.device('cpu'), torch.float32

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        per_token_feats, per_token_mask = self._get_per_token_feats()
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,
                                     batch_data_samples,
                                     per_token_feats=per_token_feats,
                                     per_token_mask=per_token_mask)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        predict_start = time.perf_counter()
        self._sync_inference_fp32_modules()
        results_list, extract_sec, head_sec = self._predict_once(
            batch_inputs,
            batch_data_samples,
            rescale=rescale,
            use_autocast=True,
        )
        low_margin_fp32_rerun = self._should_rerun_low_margin_fp32(results_list)
        if low_margin_fp32_rerun:
            results_list, extract_sec, head_sec = self._predict_once(
                batch_inputs,
                batch_data_samples,
                rescale=rescale,
                use_autocast=False,
            )

        sample_start = time.perf_counter()
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        add_sample_sec = time.perf_counter() - sample_start
        self._cache_profile_calls += 1
        if (self.text_feature_cache is not None
                and self.text_feature_cache.should_log(self._cache_profile_calls)):
            self._write_cache_debug({
                'event': 'predict_timing',
                'call': self._cache_profile_calls,
                'batch_size': int(batch_inputs.shape[0]),
                'extract_sec': round(extract_sec, 6),
                'head_sec': round(head_sec, 6),
                'add_sample_sec': round(add_sample_sec, 6),
                'low_margin_fp32_rerun': low_margin_fp32_rerun,
                'low_margin_threshold': self._low_margin_fp32_rerun_threshold,
                'total_sec': round(time.perf_counter() - predict_start, 6),
            })
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        self.texts = texts
        self._reparameterized_text_feats = None
        self._reparameterized_text_masks = None
        self._reparameterized_per_token_feats = None
        self._reparameterized_per_token_mask = None

        reparam_start = time.perf_counter()
        cached = self._lookup_reparameterize_text_cache(texts)
        if cached is not None:
            self.text_feats = cached.pooled_text_features
            self._reparameterized_text_masks = cached.text_mask
            self._reparameterized_per_token_feats = cached.token_features[:, 0]
            if cached.token_attention_mask is not None:
                self._reparameterized_per_token_mask = (
                    cached.token_attention_mask[:, 0])
            if self.text_feature_cache.should_log(self._cache_profile_calls + 1):
                self._write_cache_debug({
                    'event': 'reparameterize_cache_hit',
                    'batch_size': len(texts),
                    'total_sec': round(time.perf_counter() - reparam_start, 6),
                    'txt_feats_shape': list(self.text_feats.shape),
                })
        else:
            with self._inference_autocast_context('reparameterize'):
                text_outputs = self.backbone.forward_text(texts)
            self.text_feats, self._reparameterized_text_masks = (
                self._split_text_outputs(text_outputs))
            text_model = getattr(self.backbone, 'text_model', None)
            if text_model is not None:
                self._reparameterized_per_token_feats = getattr(
                    text_model, '_last_per_token', None)
                if self._reparameterized_per_token_feats is None:
                    self._reparameterized_per_token_feats = getattr(
                        text_model, '_per_token_feats', None)
                self._reparameterized_per_token_mask = getattr(
                    text_model, '_last_per_token_attention_mask', None)
                if self._reparameterized_per_token_mask is None:
                    self._reparameterized_per_token_mask = getattr(
                        text_model, '_per_token_mask', None)
            if (self.text_feature_cache is not None
                    and self.text_feature_cache.should_log(
                        self._cache_profile_calls + 1)):
                self._write_cache_debug({
                    'event': 'reparameterize_live',
                    'batch_size': len(texts),
                    'total_sec': round(time.perf_counter() - reparam_start, 6),
                    'txt_feats_shape': list(self.text_feats.shape),
                })
        self._reparameterized_text_feats = self.text_feats
        self._refresh_folded_attn(
            self._reparameterized_text_feats,
            self._reparameterized_per_token_feats,
            self._reparameterized_per_token_mask,
            texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        self._sync_inference_fp32_modules()
        with self._inference_autocast_context():
            img_feats, txt_feats, txt_masks = self.extract_feat(
                batch_inputs, batch_data_samples)
            per_token_feats, per_token_mask = self._get_per_token_feats()
            results = self.bbox_head.forward(img_feats, txt_feats, txt_masks,
                                             per_token_feats=per_token_feats,
                                             per_token_mask=per_token_mask)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features and optionally guide the Neck with reference tokens."""
        extract_start = time.perf_counter()
        txt_feats = None
        txt_masks = None
        cache_used = False
        cache_lookup_sec = 0.0
        image_forward_sec = 0.0
        text_forward_sec = 0.0
        neck_sec = 0.0
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
            if self._texts_match(getattr(self, 'texts', None), texts):
                txt_feats = self._reparameterized_text_feats
                txt_masks = self._reparameterized_text_masks
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
            if self._texts_match(getattr(self, 'texts', None), texts):
                txt_feats = self._reparameterized_text_feats
                txt_masks = self._reparameterized_text_masks
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        self._cached_per_token_feats = None
        self._cached_per_token_mask = None
        if txt_feats is not None:
            self._cached_per_token_feats = self._reparameterized_per_token_feats
            self._cached_per_token_mask = self._reparameterized_per_token_mask
            self._refresh_folded_attn(
                txt_feats, self._cached_per_token_feats,
                self._cached_per_token_mask, texts)
            image_start = time.perf_counter()
            img_feats = self.backbone.forward_image(batch_inputs)
            image_forward_sec = time.perf_counter() - image_start
        else:
            cache_start = time.perf_counter()
            cached = self._lookup_text_cache(texts, batch_data_samples, batch_inputs)
            cache_lookup_sec = time.perf_counter() - cache_start
            if cached is not None:
                cache_used = True
                txt_feats = cached.pooled_text_features
                txt_masks = cached.text_mask
                self._cached_per_token_feats = cached.token_features[:, 0]
                if cached.token_attention_mask is not None:
                    self._cached_per_token_mask = cached.token_attention_mask[:, 0]
                self._refresh_folded_attn(
                    txt_feats, self._cached_per_token_feats,
                    self._cached_per_token_mask, texts)
                image_start = time.perf_counter()
                img_feats = self.backbone.forward_image(batch_inputs)
                image_forward_sec = time.perf_counter() - image_start
            else:
                text_start = time.perf_counter()
                img_feats, text_outputs = self.backbone(batch_inputs, texts)
                txt_feats, txt_masks = self._split_text_outputs(text_outputs)
                self._refresh_folded_attn(
                    txt_feats, *self._get_per_token_feats(), texts)
                text_forward_sec = time.perf_counter() - text_start
        if self.with_neck:
            neck_start = time.perf_counter()
            if self.mm_neck:
                neck_guide = txt_feats
                neck_guide_mask = None
                if self.neck_guide_positive_only and txt_feats is not None:
                    per_token, per_mask = self._get_per_token_feats()
                    if per_token is not None:
                        neck_guide = per_token
                        neck_guide_mask = per_mask
                    elif txt_feats.dim() >= 2 and txt_feats.shape[1] > 1:
                        neck_guide = txt_feats[:, :1]
                if self._use_fp32_module('neck_ref_csp'):
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        img_feats = self.neck(
                            self._cast_tensor_tree_fp32(img_feats),
                            self._cast_tensor_tree_fp32(neck_guide),
                            neck_guide_mask)
                else:
                    img_feats = self.neck(img_feats, neck_guide, neck_guide_mask)
            else:
                if self._use_fp32_module('neck_ref_csp'):
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        img_feats = self.neck(
                            self._cast_tensor_tree_fp32(img_feats))
                else:
                    img_feats = self.neck(img_feats)
            neck_sec = time.perf_counter() - neck_start
        next_call = self._cache_profile_calls + 1
        if (self.text_feature_cache is not None
                and self.text_feature_cache.should_log(next_call)):
            self._write_cache_debug({
                'event': 'extract_timing',
                'cache_used': cache_used,
                'batch_size': int(batch_inputs.shape[0]),
                'cache_lookup_sec': round(cache_lookup_sec, 6),
                'image_forward_sec': round(image_forward_sec, 6),
                'text_forward_sec': round(text_forward_sec, 6),
                'neck_sec': round(neck_sec, 6),
                'total_sec': round(time.perf_counter() - extract_start, 6),
                'txt_feats_shape': None if txt_feats is None else list(txt_feats.shape),
                'per_token_shape': None if self._cached_per_token_feats is None else list(self._cached_per_token_feats.shape),
            })
        return img_feats, txt_feats, txt_masks
