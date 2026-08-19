"""Eval-only folded attention helpers for RefContrastiveHead.

The helpers keep the original training graph untouched. Fold states are built
from fixed text features after reparameterize/cache lookup and consumed only by
explicit folded forward branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FoldedHeadParams:
    """Precomputed tensors for reference-expression scoring in the head."""

    score_weight: Tensor  # (B, nh, C, L)
    score_bias: Tensor  # (B, nh, L)
    value_contribution: Tensor  # (B, nh, L)
    token_mask: Optional[Tensor]


@dataclass
class FoldedTokenAwareParams:
    """Precomputed text-side tensors for TokenAwareMaxSigmoidAttnBlock."""

    guide_q: Tensor  # (B, L, nh, hd)
    guide_v: Tensor  # (B, L, nh, hd)
    guide_mask: Optional[Tensor]


def _slice_valid_tokens(tokens: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
    """Drop right-side padding when every sample has the same valid prefix."""
    if mask is None or mask.numel() == 0:
        return tokens, mask
    valid = mask.to(dtype=torch.bool)
    max_len = int(valid.sum(dim=1).max().item())
    max_len = max(max_len, 1)
    return tokens[:, :max_len], mask[:, :max_len]


def fold_cross_attn_head(
    head: nn.Module,
    per_token_feats: Tensor,
    per_token_mask: Optional[Tensor],
    reference_text: Tensor,
) -> FoldedHeadParams:
    """Build exact tensors conditioned on the input referring expression."""
    tokens, token_mask = _slice_valid_tokens(per_token_feats, per_token_mask)
    tokens = tokens.detach()
    reference_text = reference_text.detach()
    B, L, C = tokens.shape
    nh = int(head.num_heads)
    hd = int(head.head_dim)
    scale = float(hd ** 0.5)

    w_norm = F.normalize(reference_text, dim=-1, p=2).reshape(B, C)

    k = head.k_proj(tokens).reshape(B, L, nh, hd).permute(0, 2, 1, 3)
    v = head.v_proj(tokens).reshape(B, L, nh, hd).permute(0, 2, 1, 3)

    q_weight = head.q_proj.weight.reshape(nh, hd, C)
    q_bias = head.q_proj.bias.reshape(nh, hd)
    score_weight = torch.einsum('hdc,bhld->bhcl', q_weight, k) / scale
    score_bias = torch.einsum('hd,bhld->bhl', q_bias, k) / scale

    out_weight = head.out_proj.weight
    out_bias = head.out_proj.bias
    out_to_w = torch.matmul(w_norm, out_weight).reshape(B, nh, hd)
    value_contribution = torch.einsum('bhld,bhd->bhl', v, out_to_w)
    if out_bias is not None:
        bias_score = torch.matmul(w_norm, out_bias).reshape(B, 1, 1)
        value_contribution = value_contribution + bias_score / nh
    value_contribution = (
        value_contribution
        * head.scale_context.detach().reshape(1, 1, 1))

    return FoldedHeadParams(
        score_weight=score_weight.detach(),
        score_bias=score_bias.detach(),
        value_contribution=value_contribution.detach(),
        token_mask=None if token_mask is None else token_mask.detach(),
    )


def folded_head_reference_score(
    x_flat: Tensor,
    params: FoldedHeadParams,
    store_attn: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Return the folded reference-context contribution and attention."""
    weight = params.score_weight.to(device=x_flat.device, dtype=x_flat.dtype)
    bias = params.score_bias.to(device=x_flat.device, dtype=x_flat.dtype)
    value_contribution = params.value_contribution.to(
        device=x_flat.device, dtype=x_flat.dtype)
    attn_logits = torch.einsum('bqc,bhcl->bhql', x_flat, weight)
    attn_logits = attn_logits + bias[:, :, None, :]
    if params.token_mask is not None:
        mask = params.token_mask.to(device=x_flat.device)
        pad = (1 - mask.float()).to(attn_logits.dtype)
        pad = pad.masked_fill(pad.bool(), float('-inf'))
        attn_logits = attn_logits + pad[:, None, None, :]
    attn = F.softmax(attn_logits, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    contribution = torch.einsum(
        'bhql,bhl->bq', attn, value_contribution)
    return contribution, (attn if store_attn else None)


def fold_token_aware_block(
    block: nn.Module,
    guide: Tensor,
    guide_mask: Optional[Tensor],
) -> FoldedTokenAwareParams:
    """Precompute text-side projections for TokenAwareMaxSigmoidAttnBlock."""
    tokens, token_mask = _slice_valid_tokens(guide, guide_mask)
    guide_q = block.guide_fc(tokens).reshape(
        tokens.shape[0], -1, block.num_heads, block.head_channels)
    guide_v = block.guide_value_fc(tokens).reshape(
        tokens.shape[0], -1, block.num_heads, block.head_channels)
    return FoldedTokenAwareParams(
        guide_q=guide_q.detach(),
        guide_v=guide_v.detach(),
        guide_mask=None if token_mask is None else token_mask.detach(),
    )
