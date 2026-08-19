# RefCOCO+ finetuning config based on yolo_world_v2_x

_base_ = './finetune_refcoco.py'

# RefCOCO+ dataset
refcoco_plus_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcoco+/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline={{_base_.train_pipeline}})

train_dataloader = dict(
    batch_size={{_base_.train_batch_size_per_gpu}},
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(_delete_=True, **refcoco_plus_train_dataset))

refcoco_plus_val_dataset = dict(
    type='CustomRefCOCOExpressionDataset',
    data_root='data/',
    ann_file='refcoco/refcoco+/validation.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    test_mode=True,
    pipeline={{_base_.test_pipeline}})

val_dataloader = dict(dataset=dict(_delete_=True, **refcoco_plus_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CustomRefCOCOMetric',
    ann_file='data/refcoco/refcoco+/validation.json',
    iou_threshold=0.5,
    expected_num_expressions=10758)

test_evaluator = val_evaluator
