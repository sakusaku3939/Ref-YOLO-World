# Ref-Text Contrastive Head scale_context ablation on RefCOCOg.
# Same seed and RefCOCOg train/validation setup as the full model.

_base_ = '../ref_yolo_world_refcocog.py'

randomness = dict(seed=782, diff_rank_seed=False)

work_dir = "./work_dirs/yolo_world_v2_x_finetune_refcocog_scale_context_learnable_20e_seed782"
