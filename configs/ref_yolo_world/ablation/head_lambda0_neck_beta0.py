# Ref-YOLO-World structural ablation on RefCOCOg.
# Head lambda=0 and Neck beta=0, both fixed.

_base_ = '../ref_yolo_world_refcocog.py'

randomness = dict(seed=782, diff_rank_seed=False)

work_dir = "./work_dirs/yolo_world_v2_x_finetune_refcocog_head_lambda0_neck_beta0_20e_seed782"

model = dict(
    neck=dict(
        block_cfg=dict(
            type="TokenAwareMaxSigmoidCSPLayerWithTwoConv",
            fixed_scale_context=0.0)),
    bbox_head=dict(
        head_module=dict(
            fixed_scale_context=0.0)))
