# Token-Aware Cross-Attention Block
# Implemented as separate copies of MaxSigmoidAttnBlock / MaxSigmoidCSPLayerWithTwoConv,
# leaving the original classes unchanged.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv

from ref_yolo_world.modules.folded_ref_attn import (
    FoldedTokenAwareParams,
    fold_token_aware_block,
)


@MODELS.register_module()
class TokenAwareMaxSigmoidAttnBlock(BaseModule):
    """Token-Aware Cross-Attention Block.

    A copy of MaxSigmoidAttnBlock that aggregates with softmax weights instead of
    max(dim=-1), preserving token-level information. guide is (B, N, guide_channels), where
    N is the number of tokens (L for per-token features, 1 for pooled features).

    When use_spatial_pe=True, a learnable spatial positional encoding is added to the
    attention query (image embed). Since its output layer is zero-initialized, the output
    at the start of training is identical to that without PE.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True,
                 use_spatial_pe: bool = False,
                 learnable_temperature: bool = False,
                 guide_mlp: bool = False,
                 fixed_scale_context: Optional[float] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum
        self.use_spatial_pe = use_spatial_pe
        self.learnable_temperature = learnable_temperature

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        if guide_mlp:
            self.guide_fc = nn.Sequential(
                Linear(guide_channels, embed_channels),
                nn.GELU(),
                Linear(embed_channels, embed_channels))
        else:
            self.guide_fc = Linear(guide_channels, embed_channels)
        self.guide_value_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        if learnable_temperature:
            self.temperature = nn.Parameter(
                torch.tensor(float(self.head_channels ** 0.5)))

        if use_spatial_pe:
            self.pe_proj = nn.Sequential(
                nn.Linear(2, embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels))
            nn.init.zeros_(self.pe_proj[-1].weight)
            nn.init.zeros_(self.pe_proj[-1].bias)

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)
        self.context_proj = ConvModule(
            embed_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if fixed_scale_context is None:
            self.scale_context = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer(
                "scale_context",
                torch.tensor(float(fixed_scale_context)),
                persistent=True)
        self._folded_attn_params: Optional[FoldedTokenAwareParams] = None
        self._spatial_pe_cache: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def init_weights(self):
        super().init_weights()
        # Zero-initialize the context path so that out = x_gated at the start of training
        nn.init.zeros_(self.guide_value_fc.weight)
        if self.guide_value_fc.bias is not None:
            nn.init.zeros_(self.guide_value_fc.bias)
        if hasattr(self.context_proj, 'conv'):
            nn.init.zeros_(self.context_proj.conv.weight)
            if self.context_proj.conv.bias is not None:
                nn.init.zeros_(self.context_proj.conv.bias)
        else:
            for m in self.context_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if self.use_spatial_pe:
            nn.init.zeros_(self.pe_proj[-1].weight)
            nn.init.zeros_(self.pe_proj[-1].bias)

    def set_folded_attn(
            self, guide: Optional[Tensor],
            guide_mask: Optional[Tensor] = None) -> None:
        if guide is None:
            self._folded_attn_params = None
            return
        self._folded_attn_params = fold_token_aware_block(self, guide, guide_mask)

    def _make_spatial_pe(self, H: int, W: int, embed: Tensor) -> Tensor:
        """Build a PE from normalized (x, y) coordinates, shaped to match embed."""
        gy = torch.linspace(0.0, 1.0, H, device=embed.device, dtype=embed.dtype)
        gx = torch.linspace(0.0, 1.0, W, device=embed.device, dtype=embed.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        pe = self.pe_proj(coords)  # (H, W, C)
        pe = pe.permute(2, 0, 1)  # (C, H, W)
        pe = pe.reshape(1, self.num_heads, self.head_channels, H, W)
        return pe

    def _make_spatial_pe_cached(self, H: int, W: int, embed: Tensor) -> Tensor:
        key = (H, W, embed.device, embed.dtype)
        cached = self._spatial_pe_cache.get(key)
        if cached is None:
            cached = self._make_spatial_pe(H, W, embed).detach()
            self._spatial_pe_cache[key] = cached
        return cached

    def forward(self, x: Tensor, guide: Tensor, guide_mask: Optional[Tensor] = None) -> Tensor:
        B, _, H, W = x.shape

        folded_params = self._folded_attn_params
        use_folded = folded_params is not None and folded_params.guide_q.shape[0] == B
        if use_folded:
            guide_q = folded_params.guide_q.to(device=x.device, dtype=x.dtype)
            guide_v = folded_params.guide_v.to(device=x.device, dtype=x.dtype)
            guide_mask = folded_params.guide_mask
        else:
            guide_q = self.guide_fc(guide)
            guide_v = self.guide_value_fc(guide)
            guide_q = guide_q.reshape(B, -1, self.num_heads, self.head_channels)
            guide_v = guide_v.reshape(B, -1, self.num_heads, self.head_channels)

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_spatial_pe:
            if use_folded:
                embed = embed + self._make_spatial_pe_cached(H, W, embed)
            else:
                embed = embed + self._make_spatial_pe(H, W, embed)

        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide_q)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide_q.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide_q_perm = guide_q.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide_q_perm)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        if self.learnable_temperature:
            attn_weight = attn_weight / self.temperature.clamp(min=1.0)
        else:
            attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None, None]
        gate = attn_weight.mean(dim=-1).sigmoid() * self.scale
        if guide_mask is not None and guide_mask.dim() == 2:
            m = (1 - guide_mask).to(attn_weight.dtype).to(attn_weight.device)
            m = m.masked_fill(m.bool(), float('-inf'))
            attn_weight = attn_weight + m[:, None, None, None, :]
        token_attn = F.softmax(attn_weight, dim=-1)
        token_attn = torch.nan_to_num(token_attn, nan=0.0, posinf=0.0, neginf=0.0)

        if getattr(self, 'store_token_attn', False):
            self._last_token_attn = token_attn.detach().clone()

        if self.use_einsum:
            context = torch.einsum('bmhwn,bnmc->bmchw', token_attn, guide_v)
            context = context.reshape(B, -1, H, W)
        else:
            batch, m, height, width, n = token_attn.shape
            token_attn_flat = token_attn.reshape(batch, m, -1, n)
            guide_v_flat = guide_v.permute(0, 2, 1, 3).reshape(batch, m, n, -1)
            context = torch.matmul(token_attn_flat, guide_v_flat)
            context = context.reshape(batch, m, self.head_channels, height, width)
            context = context.reshape(batch, -1, height, width)

        x_proj = self.project_conv(x)
        x_proj = x_proj.reshape(B, self.num_heads, -1, H, W)
        x_proj = x_proj * gate.unsqueeze(2)
        x_proj = x_proj.reshape(B, -1, H, W)

        context = self.context_proj(context)
        out = x_proj + self.scale_context * context
        return out


@MODELS.register_module()
class TokenAwareMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Token-Aware CSP layer.

    A copy of MaxSigmoidCSPLayerWithTwoConv that uses TokenAwareMaxSigmoidAttnBlock as
    attn_block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 num_heads: int = 1,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 with_scale: bool = False,
                 add_identity: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True,
                 use_spatial_pe: bool = False,
                 learnable_temperature: bool = False,
                 guide_mlp: bool = False,
                 fixed_scale_context: Optional[float] = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = TokenAwareMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum,
            use_spatial_pe=use_spatial_pe,
            learnable_temperature=learnable_temperature,
            guide_mlp=guide_mlp,
            fixed_scale_context=fixed_scale_context)

    def set_folded_attn(
            self, guide: Optional[Tensor],
            guide_mask: Optional[Tensor] = None) -> None:
        if hasattr(self.attn_block, "set_folded_attn"):
            self.attn_block.set_folded_attn(guide, guide_mask)

    def forward(self, x: Tensor, guide: Tensor, guide_mask: Optional[Tensor] = None) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide, guide_mask))
        return self.final_conv(torch.cat(x_main, 1))
