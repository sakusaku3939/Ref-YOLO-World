# Eval-only attention reparameterization path.
# Uses the committed standard multi-dataset spatial-flip config as a parseable
# base and enables text cache + attention reparameterization for evaluation.

_base_ = '../ref_yolo_world_multidataset.py'

text_cache_path = 'results/text_cache/refcocog_val_text_lora_features.pt'
work_dir = './work_dirs/yolo_world_v2_x_finetune_refcocog_text_attention_reparameterization_eval'

model = dict(
    use_attention_reparameterization=True,
    use_attention_reparameterization_cache=True,
    attention_reparameterization_cache_max_entries=8192,
    text_feature_cache=dict(
        path=text_cache_path,
        strict=True,
        allow_text_fallback=True,
        resident_device='cuda',
        cache_dtype='float16',
        pin_memory=False,
        profile_cache=True,
        debug_log_path='results/folded_attn/cache_debug.jsonl',
        debug_first_batches=16,
        debug_log_interval=500))

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

val_dataloader = dict(dataset=dict(_delete_=True, **refcocog_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    dump_results=True,
    expected_num_expressions=4896,
    results_dir='results/folded_attn/formal_eval')
test_evaluator = val_evaluator
