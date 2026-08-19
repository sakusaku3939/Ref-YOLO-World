# Reparameterize cache + forward-only AMP path for merged text-LoRA checkpoint.
#
# Base this eval config on the exact standard evaluation config so bbox decode,
# postprocess, and validation settings stay aligned with the checkpoint. Only the
# text backbone is swapped from LoRA to non-LoRA because the adapter weights are
# already merged into the checkpoint produced by tools/convert/merge_text_lora.py.

_base_ = '../ref_yolo_world_multidataset_text_lora.py'

work_dir = './work_dirs/yolo_world_v2_x_finetune_refcocog_text_reparam_cache_low_margin_fp32_lora_merged_eval'

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            frozen_modules=['all'],
            guide_channels=512,
            model_name='openai/clip-vit-base-patch32',
            type='HuggingCLIPLanguageBackbonePerToken',
            use_text_projection_for_per_token=True,
        )),
    inference_autocast=dict(
        enabled=True,
        dtype='float16',
        scope='forward',
        fp32_modules=['neck_ref_csp'],
        low_margin_fp32_rerun=dict(
            enabled=True,
            threshold=0.005,
        ),
    ),
    text_feature_cache=dict(
        debug_log_path='results/lora_merge/cache_debug.jsonl',
        debug_first_batches=16,
        debug_log_interval=500,
        path='results/lora_merge/refcocog_val_clip_b32_text_lora_merged_features.pt',
        strict=True,
        allow_text_fallback=True,
        resident_device='cuda',
        cache_dtype='float32',
        pin_memory=False,
        profile_cache=True,
    ))

val_evaluator = dict(
    dump_results=True,
    results_dir='results/lora_merge/formal_eval')
test_evaluator = val_evaluator
