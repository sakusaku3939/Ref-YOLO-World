# RefCOCOg + vanilla YOLO-World (fine-tune Neck + Head + reg)
# Freeze only Backbone. Train Neck / cls_preds / cls_contrasts / reg_preds.

_base_ = '../../yolo_world/finetune_refcocog.py'

max_epochs = 10
save_epoch_intervals = 1

train_cfg = dict(max_epochs=max_epochs, val_interval=save_epoch_intervals)
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        rule='greater',
        save_best='auto',
        max_keep_ckpts=-1))

# Freeze only Backbone (Neck / Head / reg_preds are trained)
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2e-4,
        weight_decay=0.05,
        batch_size_per_gpu=16),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone': dict(lr_mult=0),
            'backbone.text_model': dict(lr_mult=0),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor')
