"""Fixed 500-image LVIS smoke regression config."""

_base_ = ['./lvis_minival.py']

lvis_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root='/yolo/data/coco/',
        test_mode=True,
        ann_file='lvis/lvis_v1_minival_smoke_500_seed20260728.json',
        data_prefix=dict(img=''),
        batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline={{_base_.test_pipeline}})

test_dataloader = dict(dataset=lvis_dataset)
val_dataloader = test_dataloader
test_evaluator = dict(
    _delete_=True,
    type='mmdet.LVISMetric',
    ann_file='data/coco/lvis/lvis_v1_minival_smoke_500_seed20260728.json',
    metric='bbox')
val_evaluator = test_evaluator
