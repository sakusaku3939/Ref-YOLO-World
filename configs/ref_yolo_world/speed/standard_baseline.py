# Attention reparameterization ablation: standard Ref-YOLO-World text-LoRA baseline.

_base_ = './_speed_eval_base.py'

work_dir = './work_dirs/speed_standard_baseline'

model = dict(
    use_attention_reparameterization=False,
    use_attention_reparameterization_cache=False)

val_evaluator = dict(
    results_dir='results/attention_reparameterization_ablation/standard_baseline')
test_evaluator = val_evaluator

speed_measure = dict(
    prewarm_attention_reparameterization_cache=False)
