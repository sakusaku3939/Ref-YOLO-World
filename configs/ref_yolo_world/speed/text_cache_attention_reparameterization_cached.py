# Attention reparameterization ablation: text feature cache plus cached attention reparameterization.

_base_ = './_speed_eval_base.py'

work_dir = './work_dirs/speed_text_cache_attention_reparameterization_cached'

model = dict(
    use_attention_reparameterization=True,
    use_attention_reparameterization_cache=True,
    attention_reparameterization_cache_max_entries=8192,
    text_feature_cache=dict(
        path='results/text_cache/refcocog_val_text_lora_features.pt',
        strict=True,
        allow_text_fallback=True,
        resident_device='cuda',
        cache_dtype='float16',
        pin_memory=False,
        profile_cache=True,
        debug_log_path='results/attention_reparameterization_ablation/text_cache_attention_reparameterization_cached/cache_debug.jsonl',
        debug_first_batches=16,
        debug_log_interval=500))

val_evaluator = dict(
    results_dir='results/attention_reparameterization_ablation/text_cache_attention_reparameterization_cached/formal_eval')
test_evaluator = val_evaluator

speed_measure = dict(
    prewarm_attention_reparameterization_cache=True)
