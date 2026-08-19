# Multi-dataset + spatial flip standard baseline with CLIP text LoRA.
# This ablation starts from the standard 78.97% checkpoint and adds only
# zero-init LoRA adapters to the frozen CLIP text encoder.

_base_ = './ref_yolo_world_multidataset.py'

custom_imports = dict(
    imports=[
        'yolo_world',
        'yolo_world.evaluation.metrics',
        'yolo_world.hooks.lora_sanity_hook',
        'yolo_world.hooks.lora_sanity_hook',
        'ref_yolo_world.modules',
        'ref_yolo_world.models',
        'ref_yolo_world.modules',
        'ref_yolo_world.models',
        'ref_yolo_world.models',
    ],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        text_model=dict(
            type='HuggingCLIPLanguageBackboneLoRA',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'],
            guide_channels=512,
            use_text_projection_for_per_token=True,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.05,
            lora_target_layers=[8, 9, 10, 11],
            lora_target_modules=['q_proj', 'v_proj'],
        )))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr={{_base_.base_lr}},
        weight_decay={{_base_.weight_decay}},
        batch_size_per_gpu={{_base_.train_batch_size_per_gpu}}),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'lora_': dict(lr_mult=1.0, decay_mult=1.0),
            'logit_scale': dict(weight_decay=0.0),
        }),
    constructor='YOLOWv5OptimizerConstructor')

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='LoRASanityHook',
        expected_pairs=8,
        zero_tolerance=1e-8,
        priority='VERY_HIGH'),
]
