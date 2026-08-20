# Shared RefCOCOg validation setup for the inference-path speed ablations.

_base_ = '../ref_yolo_world_multidataset.py'

text_cache_path = 'results/text_cache/refcocog_val_text_features.pt'

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
    results_dir='results/attention_reparameterization_ablation/base')
test_evaluator = val_evaluator
