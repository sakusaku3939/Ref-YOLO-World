# Attention reparameterization ablation: attention reparameterization with per-sample text computation.

_base_ = './_speed_eval_base.py'

work_dir = './work_dirs/speed_attention_reparameterization_uncached'

model = dict(
    use_attention_reparameterization=True,
    use_attention_reparameterization_cache=False,
    attention_reparameterization_cache_max_entries=0)

val_evaluator = dict(
    results_dir='results/attention_reparameterization_ablation/attention_reparameterization_uncached/formal_eval')
test_evaluator = val_evaluator

speed_measure = dict(
    prewarm_attention_reparameterization_cache=False)
