# RefCOCOg finetuning config based on yolo_world_v2_x

_base_ = './finetune_refcoco.py'

# RefCOCOg dataset
refcocog_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcocog/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline={{_base_.train_pipeline}})

train_dataloader = dict(
    batch_size={{_base_.train_batch_size_per_gpu}},
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(_delete_=True, **refcocog_train_dataset))

refcocog_val_dataset = dict(
    type='CustomRefCOCOExpressionDataset',
    data_root='data/',
    ann_file='refcoco/refcocog/validation.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    test_mode=True,
    pipeline={{_base_.test_pipeline}})

val_dataloader = dict(dataset=dict(_delete_=True, **refcocog_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CustomRefCOCOMetric',
    ann_file='data/refcoco/refcocog/validation.json',
    iou_threshold=0.5,
    expected_num_expressions=4896)

test_evaluator = val_evaluator
