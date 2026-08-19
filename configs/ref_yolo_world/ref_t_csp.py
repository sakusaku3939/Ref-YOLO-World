# RefCOCOg + Token-Aware Neck + per-token (B, L, 512)
# Token-Aware ブロック + HuggingCLIPLanguageBackbonePerToken 使用

_base_ = '../yolo_world/finetune_refcocog.py'

custom_imports = dict(
    imports=[
        'yolo_world',
        'yolo_world.evaluation.metrics',
        'yolo_world.hooks.lora_sanity_hook',
        'ref_yolo_world.modules',
        'ref_yolo_world.models',
        'ref_yolo_world.models',
    ],
    allow_failed_imports=False)

model = dict(
    type='RefYOLOWorldDetector',
    neck_guide_positive_only=True,
    neck=dict(type='RefYOLOWorldPAFPN', block_cfg=dict(type='TokenAwareMaxSigmoidCSPLayerWithTwoConv')),
    backbone=dict(
        text_model=dict(
            type='HuggingCLIPLanguageBackbonePerToken',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'],
            guide_channels=512,
            use_text_projection_for_per_token=True,
        )))
