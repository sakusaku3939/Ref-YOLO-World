# CLIP text backbone with per-token outputs
# Returns (B, 80, 512) for the Head and stores per-token features (B, L, 512) of the single
# input referring expression in _last_per_token for the Neck

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.utils import OptMultiConfig
from transformers import AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for a frozen linear projection."""

    def __init__(self,
                 linear: nn.Linear,
                 rank: int = 16,
                 alpha: float = 32.0,
                 dropout: float = 0.05) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f'rank must be positive, got {rank}')

        self.linear = linear
        self.rank = rank
        self.lora_alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def merge_weights(self) -> None:
        """Merge LoRA delta into the wrapped linear projection for inference."""
        with torch.no_grad():
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)


@MODELS.register_module()
@MMDET_MODELS.register_module()
class HuggingCLIPLanguageBackbonePerToken(BaseModule):
    """CLIP text backbone: pooled (B, 80, 512) for the Head + per-token (B, L, 512) for the Neck.

    forward() returns (txt_feats, text_mask) as in HuggingCLIPLanguageBackbone, so the Head
    still receives (B, 80, 512).
    Per-token features of the single input referring expression are stored in
    self._last_per_token (B, L, 512), and the Detector passes them to the Neck when
    neck_guide_positive_only is set.
    """

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 add_mask: bool = False,
                 training_use_cache: bool = False,
                 guide_channels: int = 512,
                 use_text_projection_for_per_token: bool = True,
                 per_token_mlp: bool = False,
                 use_penultimate_for_per_token: bool = False,
                 mask_special_tokens: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.guide_channels = guide_channels
        self.use_text_projection_for_per_token = use_text_projection_for_per_token
        self.per_token_mlp = per_token_mlp
        self.use_penultimate_for_per_token = use_penultimate_for_per_token
        self.mask_special_tokens = mask_special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(
            model_name, attention_dropout=dropout, output_hidden_states=True)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self.pad_value = 0

        hidden_size = getattr(
            getattr(self.model.config, 'text_config', None), 'hidden_size',
            getattr(self.model.config, 'hidden_size', 512))
        if per_token_mlp:
            self.per_token_proj = nn.Sequential(
                nn.Linear(hidden_size, guide_channels),
                nn.GELU(),
                nn.Linear(guide_channels, guide_channels))
        elif not use_text_projection_for_per_token and hidden_size != guide_channels:
            self.per_token_proj = nn.Linear(hidden_size, guide_channels)
        else:
            self.per_token_proj = None

        self._freeze_modules()
        self._last_per_token: Optional[Tensor] = None

    def forward(self, text: List[List[str]]) -> Tuple[Tensor, Optional[Tensor]]:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        n_slots = num_per_batch[0]
        text_flat = list(itertools.chain(*text))
        if self.add_mask:
            text_mask = torch.tensor(
                [x != self.pad_value for x in text_flat],
                requires_grad=False).to(self.model.device)
        text_tokens = self.tokenizer(
            text=text_flat, return_tensors='pt', padding=True)
        text_tokens = text_tokens.to(device=self.model.device)

        if len(self.frozen_modules) > 0:
            with torch.no_grad():
                txt_outputs = self.model(**text_tokens)
                txt_feats = txt_outputs.text_embeds
        else:
            txt_outputs = self.model(**text_tokens)
            txt_feats = txt_outputs.text_embeds

        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, n_slots, txt_feats.shape[-1])
        if self.add_mask:
            text_mask = text_mask.reshape(-1, n_slots).to(txt_feats)
        else:
            text_mask = None

        # Per-token features of the single input referring expression (B, L, guide_channels)
        # use_text_projection_for_per_token=True: CLIP text_projection (same space as pooled)
        # use_text_projection_for_per_token=False: last_hidden_state only, or per_token_proj to
        # match dimensions
        # Hidden states for per-token features (last or penultimate layer)
        hidden_states = getattr(txt_outputs, 'hidden_states', None)
        if hidden_states is not None:
            if self.use_penultimate_for_per_token and len(hidden_states) >= 2:
                last_hidden = hidden_states[-2]
            else:
                last_hidden = hidden_states[-1]
        else:
            last_hidden = getattr(txt_outputs, 'last_hidden_state', None)
        if last_hidden is not None and n_slots > 0:
            B = txt_feats.shape[0]
            seq_len = last_hidden.shape[1]
            hidden_size = last_hidden.shape[2]
            first_sent = last_hidden.view(B, n_slots, seq_len, hidden_size)[:, 0]
            if self.per_token_mlp:
                first_sent = self.per_token_proj(first_sent)
            elif self.use_text_projection_for_per_token:
                text_proj = self.model.text_projection
                first_sent = text_proj(first_sent)
            elif self.per_token_proj is not None:
                first_sent = self.per_token_proj(first_sent)
            first_sent = first_sent / first_sent.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            self._last_per_token = first_sent
            # attention_mask: 1=valid, 0=padding -> (B, L) for the first sentence only
            if 'attention_mask' in text_tokens:
                am = text_tokens['attention_mask']
                am = am.view(B, n_slots, -1)[:, 0]

                # Mask special tokens such as BOS / EOS
                if self.mask_special_tokens and 'input_ids' in text_tokens:
                    ids = text_tokens['input_ids'].view(B, n_slots, -1)[:, 0]
                    bos_id = self.tokenizer.bos_token_id
                    eos_id = self.tokenizer.eos_token_id
                    special_mask = torch.zeros_like(ids, dtype=torch.bool)
                    if bos_id is not None:
                        special_mask |= (ids == bos_id)
                    if eos_id is not None:
                        special_mask |= (ids == eos_id)
                    # attention_mask: 1=valid, 0=padding/special
                    am = am.masked_fill(special_mask, 0)

                self._last_per_token_attention_mask = am.to(first_sent.device)
            else:
                self._last_per_token_attention_mask = None
        else:
            self._last_per_token = None
            self._last_per_token_attention_mask = None

        return txt_feats, text_mask

    def _freeze_modules(self) -> None:
        if len(self.frozen_modules) == 0:
            return
        if self.frozen_modules[0] == 'all':
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode: bool = True) -> 'HuggingCLIPLanguageBackbonePerToken':
        super().train(mode)
        self._freeze_modules()
        return self


@MODELS.register_module()
@MMDET_MODELS.register_module()
class HuggingCLIPLanguageBackboneLoRA(HuggingCLIPLanguageBackbonePerToken):
    """Per-token CLIP text backbone with LoRA on frozen self-attention Q/V.

    The parent backbone wraps the frozen CLIP forward in ``torch.no_grad()``.
    This subclass keeps the base CLIP parameters frozen but bypasses that
    no-grad block so gradients can reach the injected LoRA matrices.
    """

    def __init__(self,
                 *args,
                 lora_target_layers: Sequence[int] = (8, 9, 10, 11),
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Sequence[str] = ('q_proj', 'v_proj'),
                 **kwargs) -> None:
        kwargs['frozen_modules'] = ['all']
        super().__init__(*args, **kwargs)

        self.lora_target_layers = list(lora_target_layers)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout
        self.lora_target_modules = list(lora_target_modules)
        self._inject_lora()

    def _inject_lora(self) -> None:
        encoder_layers = self.model.text_model.encoder.layers
        for layer_idx in self.lora_target_layers:
            layer = encoder_layers[layer_idx]
            self_attn = layer.self_attn
            for module_name in self.lora_target_modules:
                original_linear = getattr(self_attn, module_name)
                if isinstance(original_linear, LoRALinear):
                    continue
                setattr(
                    self_attn,
                    module_name,
                    LoRALinear(
                        original_linear,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout_rate))
        self._enable_lora_grad()

    def _enable_lora_grad(self) -> None:
        encoder_layers = self.model.text_model.encoder.layers
        for layer_idx in self.lora_target_layers:
            layer = encoder_layers[layer_idx]
            self_attn = layer.self_attn
            for module_name in self.lora_target_modules:
                module = getattr(self_attn, module_name)
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)

    def _freeze_modules(self) -> None:
        super()._freeze_modules()
        if hasattr(self, 'lora_target_layers'):
            self._enable_lora_grad()

    def forward(self, text: List[List[str]]) -> Tuple[Tensor, Optional[Tensor]]:
        saved_frozen = self.frozen_modules
        self.frozen_modules = []
        try:
            return super().forward(text)
        finally:
            self.frozen_modules = saved_frozen

    def merge_lora_weights(self) -> None:
        encoder_layers = self.model.text_model.encoder.layers
        for layer_idx in self.lora_target_layers:
            layer = encoder_layers[layer_idx]
            self_attn = layer.self_attn
            for module_name in self.lora_target_modules:
                module = getattr(self_attn, module_name)
                if isinstance(module, LoRALinear):
                    module.merge_weights()
