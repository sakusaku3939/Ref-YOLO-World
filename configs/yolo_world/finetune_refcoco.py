# RefCOCO finetuning config based on yolo_world_v2_x

_base_ = './yolo_world_v2_x.py'

# hyper-parameters for RefCOCO
num_classes = 1
# Keep the pretrained 80-slot training interface. RefCOCO supplies one input
# referring expression; the remaining slots are empty-string padding, not
# sampled semantic negative expressions.
num_training_classes = 80
max_epochs = 20
save_epoch_intervals = 1
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = 'weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth'

# model settings - override num_test_classes for RefCOCO
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings (pre_transform / last_transform for configs that extend pipeline e.g. KD)
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform
text_transform = [
    dict(type='RandomLoadText',
         # The dataset exposes one text class, so no semantic negatives can be
         # sampled here. padding_to_max only preserves the fixed-width layout.
         num_neg_samples=(0, 80),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100.,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform
]

refcoco_train_dataset = dict(
    type='CustomRefCOCODataset',
    data_root='data/',
    ann_file='refcoco/refcoco/train.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True, **refcoco_train_dataset))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

refcoco_val_dataset = dict(
    type='CustomRefCOCOExpressionDataset',
    data_root='data/',
    ann_file='refcoco/refcoco/validation.json',
    img_prefix='data/coco/train2014/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    test_mode=True,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dict(_delete_=True, **refcoco_val_dataset))
test_dataloader = val_dataloader

# training settings
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        rule='greater',
        save_best='auto',
        max_keep_ckpts=-1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='CustomRefCOCOMetric',
    ann_file='data/refcoco/refcoco/validation.json',
    iou_threshold=0.5,
    expected_num_expressions=10834)

test_evaluator = val_evaluator
