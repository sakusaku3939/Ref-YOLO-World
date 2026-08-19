# Ref-YOLO-World: Cross-Attention Contrastive Head for Referring Expression Comprehension

from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmcv.cnn.bricks import build_norm_layer
from mmyolo.registry import MODELS

from ref_yolo_world.modules.folded_ref_attn import (
    FoldedHeadParams,
    folded_head_reference_score,
)


@MODELS.register_module()
class CrossAttnContrastiveHead(BaseModule):
    """Cross-Attention Contrastive Head for Referring Expression Comprehension.

    Multi-head cross-attention enriches image features with per-token text
    information before computing contrastive scores.

    Text slot 0 is the input referring expression. Image features attend to
    its tokens via cross-attention and are scored against its pooled feature.
    Additional slots use the standard BNContrastiveHead dot product. In the
    baseline RefCOCO pipeline, those training slots contain empty-string
    padding rather than sampled semantic negative expressions.

    out_proj and scale_context are zero-initialized so that at training start,
    output is numerically identical to BNContrastiveHead.
    """

    LOGIT_SCALE_MAX = 4.6052  # ln(100), following OpenAI CLIP

    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 num_heads: int = 8,
                 use_2d_positional_encoding: bool = False,
                 use_einsum: bool = True,
                 fixed_scale_context: Optional[float] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert embed_dims % num_heads == 0, \
            f'embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.use_2d_positional_encoding = use_2d_positional_encoding
        self.use_einsum = use_einsum
        self.fixed_scale_context = fixed_scale_context

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        if fixed_scale_context is None:
            self.scale_context = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer(
                'scale_context',
                torch.tensor(float(fixed_scale_context)),
                persistent=True)
        self._folded_attn_params: Optional[FoldedHeadParams] = None

        if use_2d_positional_encoding:
            self.pe_proj = nn.Sequential(
                nn.Linear(2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims))
            nn.init.zeros_(self.pe_proj[-1].weight)
            nn.init.zeros_(self.pe_proj[-1].bias)

    def init_weights(self):
        super().init_weights()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.use_2d_positional_encoding:
            nn.init.zeros_(self.pe_proj[-1].weight)
            nn.init.zeros_(self.pe_proj[-1].bias)

    def set_folded_attn_params(
            self, params: Optional[FoldedHeadParams]) -> None:
        self._folded_attn_params = params
    @staticmethod
    def _autocast_disabled_context(enabled: bool):
        if enabled and torch.cuda.is_available():
            return torch.amp.autocast(device_type='cuda', enabled=False)
        return nullcontext()

    @staticmethod
    def _use_fp32_module(
        fp32_modules: Optional[Sequence[str]],
        module_name: str,
    ) -> bool:
        return fp32_modules is not None and module_name in fp32_modules

    def _make_2d_pe(self, H: int, W: int, x: Tensor) -> Tensor:
        gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)
        pe = self.pe_proj(coords)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def _cross_attention(self,
                         x_flat: Tensor,
                         per_token_feats: Tensor,
                         per_token_mask: Optional[Tensor],
                         force_fp32: bool = False) -> Tensor:
        """Multi-head cross-attention: image (Q) x text tokens (K, V).

        Args:
            x_flat: (B, HW, C) image features.
            per_token_feats: (B, L, C) per-token text features.
            per_token_mask: (B, L), 1=valid 0=padding.
        Returns:
            (B, HW, C) context vector.
        """
        with self._autocast_disabled_context(force_fp32):
            if force_fp32:
                x_flat = x_flat.float()
                per_token_feats = per_token_feats.float()

            B, HW, C = x_flat.shape
            nh, hd = self.num_heads, self.head_dim

            q = self.q_proj(x_flat).reshape(B, HW, nh, hd).permute(0, 2, 1, 3)
            k = self.k_proj(per_token_feats).reshape(B, -1, nh, hd).permute(0, 2, 1, 3)
            v = self.v_proj(per_token_feats).reshape(B, -1, nh, hd).permute(0, 2, 1, 3)

            attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)

            if per_token_mask is not None:
                pad = (1 - per_token_mask.float()).to(attn.dtype)
                pad = pad.masked_fill(pad.bool(), float('-inf'))
                attn = attn + pad[:, None, None, :]

            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

            if getattr(self, 'store_token_attn', False):
                self._last_raw_attn = attn.detach().clone()

            ctx = torch.matmul(attn, v)
            ctx = ctx.permute(0, 2, 1, 3).reshape(B, HW, C)
            return self.out_proj(ctx)

    def _dot_score(self, x: Tensor, w_norm: Tensor) -> Tensor:
        if self.use_einsum:
            return torch.einsum('bchw,bkc->bkhw', x, w_norm)
        B, C, H, W = x.shape
        K = w_norm.shape[1]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        out = torch.matmul(x_flat, w_norm.permute(0, 2, 1))
        return out.reshape(B, H, W, K).permute(0, 3, 1, 2)

    def forward(self, x: Tensor, w: Tensor,
                per_token_feats: Optional[Tensor] = None,
                per_token_mask: Optional[Tensor] = None,
                fp32_modules: Optional[Sequence[str]] = None) -> Tensor:
        """
        Args:
            x: image features (B, C, H, W).
            w: pooled text features (B, K, C). K=num_classes.
            per_token_feats: per-token features (B, L, C) for the input
                referring expression in text slot 0.
            per_token_mask: (B, L), 1=valid 0=padding.
        Returns:
            cls_logit: (B, K, H, W).
        """
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        spatial_pe_fp32 = self._use_fp32_module(fp32_modules, 'spatial_pe')
        cross_attn_fp32 = self._use_fp32_module(fp32_modules, 'head_cross_attention')
        contrastive_fp32 = self._use_fp32_module(fp32_modules, 'head_contrastive')

        if self.use_2d_positional_encoding:
            if spatial_pe_fp32:
                with self._autocast_disabled_context(True):
                    x_norm = x_norm.float()
                    x_norm = x_norm + self._make_2d_pe(H, W, x_norm)
            else:
                x_norm = x_norm + self._make_2d_pe(H, W, x_norm)

        with self._autocast_disabled_context(contrastive_fp32):
            if contrastive_fp32:
                w_norm = F.normalize(w.float(), dim=-1, p=2)
            else:
                w_norm = F.normalize(w, dim=-1, p=2)

        folded_params = self._folded_attn_params
        use_folded = (
            folded_params is not None
            and per_token_feats is not None
            and folded_params.score_weight.shape[0] == B)
        if use_folded:
            x_flat = x_norm.flatten(2).transpose(1, 2)
            base_score = self._dot_score(x_norm, w_norm[:, :1])
            contribution, attn = folded_head_reference_score(
                x_flat,
                folded_params,
                store_attn=getattr(self, 'store_token_attn', False))
            if attn is not None:
                self._last_token_attn = attn.detach().reshape(
                    B, -1, H, W, attn.shape[-1])
            reference_score = base_score + contribution.reshape(B, 1, H, W)
        elif per_token_feats is not None:
            x_flat = x_norm.flatten(2).transpose(1, 2)
            ctx = self._cross_attention(
                x_flat,
                per_token_feats,
                per_token_mask,
                force_fp32=cross_attn_fp32)

            if getattr(self, 'store_token_attn', False) and hasattr(self, '_last_raw_attn'):
                self._last_token_attn = self._last_raw_attn.reshape(
                    B, -1, H, W, self._last_raw_attn.shape[-1])
                del self._last_raw_attn

            ctx = ctx.transpose(1, 2).reshape(B, C, H, W)
            if cross_attn_fp32 and not contrastive_fp32:
                ctx = ctx.to(dtype=x_norm.dtype)
            with self._autocast_disabled_context(cross_attn_fp32 or contrastive_fp32):
                base_x = x_norm.float() if (cross_attn_fp32 or contrastive_fp32) else x_norm
                ctx_x = ctx.float() if (cross_attn_fp32 or contrastive_fp32) else ctx
                scale = self.scale_context.float() if (cross_attn_fp32 or contrastive_fp32) else self.scale_context
                x_enriched = base_x + scale * ctx_x
            if contrastive_fp32:
                with self._autocast_disabled_context(True):
                    reference_score = self._dot_score(x_enriched.float(),
                                                      w_norm[:, :1])
            else:
                reference_score = self._dot_score(
                    x_enriched.to(dtype=w_norm.dtype),
                    w_norm[:, :1])
        else:
            if contrastive_fp32:
                with self._autocast_disabled_context(True):
                    reference_score = self._dot_score(x_norm.float(),
                                                      w_norm[:, :1])
            else:
                reference_score = self._dot_score(
                    x_norm.to(dtype=w_norm.dtype),
                    w_norm[:, :1])

        if w.shape[1] > 1:
            if contrastive_fp32:
                with self._autocast_disabled_context(True):
                    remaining_score = self._dot_score(x_norm.float(),
                                                      w_norm[:, 1:])
            else:
                remaining_score = self._dot_score(
                    x_norm.to(dtype=w_norm.dtype),
                    w_norm[:, 1:])
            cls_logit = torch.cat([reference_score, remaining_score], dim=1)
        else:
            cls_logit = reference_score

        logit_scale = self.logit_scale.clamp(max=self.LOGIT_SCALE_MAX).exp()
        return cls_logit * logit_scale + self.bias
