# RefCOCO / RefCOCO+ / RefCOCOg joint training with spatial flip text sync.
# This config is the multi-dataset + spatial-flip training path and a training base for later Ref-YOLO-World (text LoRA) speed work.
# It keeps the model, losses, postprocess, and RefCOCOg validation path fixed,
# while using the multi-dataset train distribution and horizontal-flip text
# consistency that reached the past baseline 78.97% Acc@0.5 on RefCOCOg val.

_base_ = './ref_yolo_world_refcocog.py'

standard_evaluation_checkpoint = (
    'weights/ref_yolo_world_x_multidataset_text_lora.pth')
standard_training_start_checkpoint = (
    'weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_'
    'pretrain_1280ft-14996a36.pth')
load_from = standard_training_start_checkpoint
resume = False

# The base RefCOCOg pipeline ends with RandomLoadText and PackDetInputs.
# Insert the correction after RandomFlip has populated flip metadata and before
# RandomLoadText flattens/pads the referring expression text slots.
spatial_flip_train_pipeline = [
    *_base_.train_pipeline[:-2],
    dict(type='SpatialTextFlipCorrection'),
    *_base_.train_pipeline[-2:],
]

refcoco_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcoco/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=spatial_flip_train_pipeline)

refcoco_plus_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcoco+/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=spatial_flip_train_pipeline)

refcocog_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcocog/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=spatial_flip_train_pipeline)

train_dataloader = dict(
    batch_size={{_base_.train_batch_size_per_gpu}},
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            refcoco_train_dataset,
            refcoco_plus_train_dataset,
            refcocog_train_dataset,
        ],
        ignore_keys=['classes', 'palette']))

# Keep formal validation fixed to RefCOCOg val. Future experiments using this
# standard path should compare against the current 80.37% Ref-YOLO-World (text LoRA) result; 78.97% the multi-dataset + spatial-flip baseline is a past baseline only.
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
