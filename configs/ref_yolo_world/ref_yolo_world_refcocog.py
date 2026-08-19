# Token-Aware Neck (Spatial PE) + Cross-Attention Head
# Neck: TokenAwareMaxSigmoidCSPLayerWithTwoConv (use_spatial_pe=True)
# Head: CrossAttnContrastiveHead (use_positional_encoding=False; Neck で PE 済み)

_base_ = './ref_t_csp_spatial_pe.py'

custom_imports = dict(
    imports=[
        'yolo_world',
        'yolo_world.evaluation.metrics',
        'yolo_world.hooks.lora_sanity_hook',
        'ref_yolo_world.modules',
        'ref_yolo_world.models',
        'ref_yolo_world.modules',
        'ref_yolo_world.models',
        'ref_yolo_world.models',
    ],
    allow_failed_imports=False)

model = dict(
    type='RefYOLOWorldDetector',
    neck_guide_positive_only=True,
    bbox_head=dict(
        type='RefYOLOWorldHead',
        head_module=dict(
            type='RefYOLOWorldHeadModule',
            embed_dims=512,
            num_classes=80,
            cross_attn_num_heads=8,
            use_positional_encoding=False)))
