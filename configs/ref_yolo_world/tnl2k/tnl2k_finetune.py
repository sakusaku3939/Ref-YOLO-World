"""TNL2K evaluation: sequence-level TNL2K adaptation of the Ref-YOLO-World text-LoRA model."""

_base_ = '../ref_yolo_world_multidataset_text_lora.py'

max_epochs = 3
save_epoch_intervals = 1
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16

# Probe-only adaptation start. This is intentionally not a standard-baseline
# training origin: TNL2K evaluation compares zero-shot Ref-YOLO-World against its own
# TNL2K-adapted counterpart.
load_from = (
    'weights/ref_yolo_world_x_multidataset_text_lora.pth')
resume = False

tnl2k_train_dataset = dict(
    type='TNL2KFrameDataset',
    data_root='data/TNL2K_standard',
    ann_file='manifests/train.jsonl',
    pipeline={{_base_.spatial_flip_train_pipeline}})

tnl2k_validation_dataset = dict(
    type='TNL2KFrameDataset',
    data_root='data/TNL2K_standard',
    ann_file='manifests/validation.jsonl',
    test_mode=True,
    pipeline={{_base_.test_pipeline}})

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(_delete_=True, **tnl2k_train_dataset))
val_dataloader = dict(
    dataset=dict(_delete_=True, **tnl2k_validation_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='TNL2KFrameMetric',
    ann_file='data/TNL2K_standard/manifests/validation.jsonl',
    iou_threshold=0.5)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=save_epoch_intervals)
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        rule='greater',
        save_best='tnl2k/Acc@0.5',
        max_keep_ckpts=-1))

# The released checkpoint already contains trained, non-zero LoRA B weights. The inherited
# LoRASanityHook checks zero initialization for newly injected adapters and is
# therefore not valid for this probe-only checkpoint continuation.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

work_dir = 'work_dirs/tnl2k_finetune'
