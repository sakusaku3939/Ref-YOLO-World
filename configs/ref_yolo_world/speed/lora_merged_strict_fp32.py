# Strict FP32 ablation for the LoRA-merged cache path.
#
# The cached pooled/token tensors are stored and kept resident as float32, and
# detector inference runs without autocast. This isolates text-feature and
# reparameterize caching from every FP16 speedup.

_base_ = './lora_merged_low_margin_fp32.py'

work_dir = './work_dirs/yolo_world_v2_x_finetune_refcocog_text_reparam_cache_lora_merged_strict_fp32_eval'

model = dict(
    inference_autocast=dict(
        _delete_=True,
        enabled=False,
    ),
    text_feature_cache=dict(
        path='results/lora_merge/refcocog_val_clip_b32_text_lora_merged_features.pt',
        cache_dtype='float32',
        debug_log_path='results/lora_merge_strict_fp32/cache_debug.jsonl',
    ),
)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'ann_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts')),
]

refcocog_val_dataset = dict(
    type='CustomRefCOCOExpressionDataset',
    data_root='data/',
    ann_file='refcoco/refcocog/validation.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    test_mode=True,
    pipeline=test_pipeline)

val_dataloader = dict(
    dataset=dict(_delete_=True, **refcocog_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    dump_results=True,
    expected_num_expressions=4896,
    results_dir='results/lora_merge_strict_fp32/formal_eval')
test_evaluator = val_evaluator
