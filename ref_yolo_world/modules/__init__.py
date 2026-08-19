"""Building blocks of Ref-YOLO-World.

- ``token_aware_blocks``: Ref-T-CSPLayer (per-token guided Neck block).
- ``ref_contrastive_blocks``: Ref-Text Contrastive Head with cross attention.
- ``folded_ref_attn``: attention reparameterization for cached inference.
- ``text_feature_cache``: on-disk cache of CLIP text features.
- ``text_backbones``: per-token CLIP text backbone and its LoRA variant.
- ``ref_transforms``: flip-aware referring-expression text correction.
"""

from .folded_ref_attn import (FoldedHeadParams, FoldedTokenAwareParams,
                              fold_cross_attn_head)
from .ref_contrastive_blocks import CrossAttnContrastiveHead
from .ref_transforms import SpatialTextFlipCorrection
from .text_backbones import (HuggingCLIPLanguageBackboneLoRA,
                             HuggingCLIPLanguageBackbonePerToken)
from .text_feature_cache import TextFeatureCache
from .token_aware_blocks import (TokenAwareMaxSigmoidAttnBlock,
                                 TokenAwareMaxSigmoidCSPLayerWithTwoConv)

__all__ = [
    'CrossAttnContrastiveHead',
    'FoldedHeadParams',
    'FoldedTokenAwareParams',
    'fold_cross_attn_head',
    'HuggingCLIPLanguageBackboneLoRA',
    'HuggingCLIPLanguageBackbonePerToken',
    'SpatialTextFlipCorrection',
    'TextFeatureCache',
    'TokenAwareMaxSigmoidAttnBlock',
    'TokenAwareMaxSigmoidCSPLayerWithTwoConv',
]
