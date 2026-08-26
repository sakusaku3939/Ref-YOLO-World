"""LVIS minival regression config for the Ref-YOLO-World text-LoRA model."""

_base_ = ['../ref_yolo_world_multidataset.py']

img_scale = (1280, 1280)
num_classes = 1203
model = dict(num_test_classes=num_classes)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts')),
]

lvis_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root='data/coco/',
        test_mode=True,
        ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=lvis_dataset)
val_dataloader = test_dataloader
test_evaluator = dict(
    _delete_=True,
    type='mmdet.LVISMetric',
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox')
val_evaluator = test_evaluator
