# Token-Aware Neck + Spatial Positional Encoding
# Adds a learnable PE to the image query of cross-attention
# The output layer of pe_proj is zero-initialized, so at the start of training the
# output is identical to the variant without PE

_base_ = './ref_t_csp.py'

model = dict(
    neck=dict(
        block_cfg=dict(
            type='TokenAwareMaxSigmoidCSPLayerWithTwoConv',
            use_spatial_pe=True)))
