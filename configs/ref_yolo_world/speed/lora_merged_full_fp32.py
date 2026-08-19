# Full-FP32 evaluation path for the merged text-LoRA checkpoint.
#
# This keeps the text-feature and reparameterize-state caches but
# disables inference autocast and the low-margin rerun, so every detector
# forward is measured in FP32 under the same RefCOCOg validation setup.

_base_ = './lora_merged_low_margin_fp32.py'

work_dir = './work_dirs/yolo_world_v2_x_finetune_refcocog_text_reparam_cache_lora_merged_full_fp32_eval'

model = dict(
    inference_autocast=dict(
        _delete_=True,
        enabled=False,
    ),
    text_feature_cache=dict(
        debug_log_path='results/lora_merge_full_fp32/cache_debug.jsonl',
    ),
)

val_evaluator = dict(
    dump_results=True,
    results_dir='results/lora_merge_full_fp32/formal_eval',
)
test_evaluator = val_evaluator
