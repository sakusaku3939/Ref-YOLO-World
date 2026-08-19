# Token-Aware Neck + Spatial Positional Encoding
# Cross-Attention の image query に学習可能 PE を加算
# pe_proj の出力層は 0 初期化 → 学習開始時は PE なしと完全同一

_base_ = './ref_t_csp.py'

model = dict(
    neck=dict(
        block_cfg=dict(
            type='TokenAwareMaxSigmoidCSPLayerWithTwoConv',
            use_spatial_pe=True)))
