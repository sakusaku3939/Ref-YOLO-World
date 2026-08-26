# Training and evaluation

Every command assumes the repository root as working directory and the layout
described in [data.md](data.md). Two L40S GPUs were used for all reported
numbers.

## Training

The released model is trained in one run from the YOLO-World v2-X pretrained
weights (20 epochs, best checkpoint at epoch 10):

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  2 --amp \
  --cfg-options load_from=weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
```

The intermediate configs train the same model with fewer components and are the
module ablation of the method:

| Config | Neck | Spatial PE | Contrastive head | Text LoRA | Train data |
| --- | :-: | :-: | :-: | :-: | --- |
| `configs/yolo_world/finetune_refcocog.py` | – | – | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_t_csp.py` | ✓ | – | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_t_csp_spatial_pe.py` | ✓ | ✓ | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_yolo_world_refcocog.py` | ✓ | ✓ | ✓ | – | RefCOCOg |
| `configs/ref_yolo_world/ref_yolo_world_multidataset_no_lora.py` | ✓ | ✓ | ✓ | – | RefCOCO/+/g + flip sync |
| `configs/ref_yolo_world/ref_yolo_world_multidataset.py` | ✓ | ✓ | ✓ | ✓ | RefCOCO/+/g + flip sync |

## Accuracy

### RefCOCOg val (headline number)

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  weights/ref_yolo_world_x_multidataset.pth \
  2
```

Expected: `Acc@0.5 = 81.62` over 4,896 expressions.

The configs evaluate every referring expression as one sample. The older
one-sample-per-target protocol reports `80.37` (2068 / 2573) for the same
checkpoint; the two numbers differ only in the denominator.

### All eight RefCOCO / RefCOCO+ / RefCOCOg splits

```bash
python3 tools/eval/refcoco_suite_benchmark.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --gpus 2 \
  --output-dir results/refcoco_suite_accuracy
```

### Accuracy by expression type

Generate the per-expression results, then compute the slice metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  weights/ref_yolo_world_x_multidataset.pth \
  --cfg-options \
    test_evaluator.dump_results=True \
    test_evaluator.results_dir=results/refcocog_detailed_results

python3 tools/eval/refcocog_slice_eval.py \
  --ann-file data/refcoco/refcocog/validation.json \
  --detailed-results results/refcocog_detailed_results/detailed_results.json \
  --output-json results/refcocog_slice_summary.json
```

### Ref-Adv-s

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  weights/ref_yolo_world_x_multidataset.pth \
  --cfg-options \
    test_dataloader.dataset.ann_file=ref_adv/ref_adv_s.json \
    test_evaluator.ann_file=data/ref_adv/ref_adv_s.json \
    test_evaluator.expected_num_expressions=1142
```

### LVIS minival regression

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/eval/lvis_minival.py \
  weights/ref_yolo_world_x_multidataset.pth \
  2
```

### TNL2K

Zero-shot and fine-tuned comparison, including the official MATLAB/Octave
metrics:

```bash
python3 tools/eval/run_tnl2k_experiment.py --gpus 2 --amp --octave
```

## Inference speed

The seven inference-path conditions live in `configs/ref_yolo_world/speed/`:

| Config | Text cache | Attention reparameterization | Reparameterized-state cache |
| --- | :-: | :-: | :-: |
| `standard_baseline.py` | – | – | – |
| `text_cache_only.py` | ✓ | – | – |
| `attention_reparameterization_uncached.py` | – | ✓ | – |
| `text_cache_attention_reparameterization_uncached.py` | ✓ | ✓ | – |
| `text_cache_attention_reparameterization_cached.py` | ✓ | ✓ | ✓ |
| `lora_merged_low_margin_fp32.py` | ✓ | ✓ | ✓ (FP16 forward, FP32 rerun near the decision boundary) |
| `lora_merged_full_fp32.py` | ✓ | ✓ | ✓ (FP32 everywhere) |

Measure one condition:

```bash
python3 tools/speed/reliable_speed_measure.py \
  --label text_cache_attention_reparameterization_cached \
  --config configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --ann-file data/refcoco/refcocog/validation.json \
  --output results/speed/text_cache_attention_reparameterization_cached.json
```

Expected on a single L40S: `38.26 ms/image` (26.14 FPS) for
`standard_baseline.py` and `20.08 ms/image` (49.59 FPS) for
`lora_merged_low_margin_fp32.py`, at identical accuracy.

Per-stage breakdown (backbone / neck / head / text encoder):

```bash
python3 tools/speed/stage_speed_breakdown.py \
  --ref-config configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  --ref-checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --output-dir results/speed/stage_breakdown

python3 tools/speed/measure_ref_yoloworld_params.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --output results/speed/params.json
```

## Equivalence checks

Attention reparameterization is an exact algebraic rewrite of the cross
attention. The following two checks assert that it does not change the logits:

```bash
python3 tools/eval/attention_reparameterization_equivalence.py \
  --config configs/ref_yolo_world/speed/attention_reparameterization_eval.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --ann-file data/refcoco/refcocog/validation.json \
  --output results/attention_reparameterization_equivalence.json

python3 tools/eval/reparam_cache_equivalence.py \
  --config configs/ref_yolo_world/speed/lora_merged_low_margin_fp32.py \
  --checkpoint weights/ref_yolo_world_x_lora_merged.pth \
  --ann-file data/refcoco/refcocog/validation.json \
  --output results/reparam_cache_equivalence.json
```

## Export

```bash
python3 tools/export/export_ref_yoloworld_text_cache_onnx.py \
  configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  weights/ref_yolo_world_x_multidataset.pth \
  --output work_dirs/export/ref_yolo_world.onnx

python3 tools/export/eval_ref_yoloworld_onnx_bbox_acc.py \
  configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  weights/ref_yolo_world_x_multidataset.pth \
  --onnx work_dirs/export/ref_yolo_world.onnx \
  --ann-file data/refcoco/refcocog/validation.json \
  --output results/export/ref_yolo_world_onnx_bbox_acc.json
```

## Ablations

```bash
# fine-tuning scope (which parts of YOLO-World are unfrozen)
./tools/dist_train.sh configs/ref_yolo_world/ablation/finetune_scope_neck_only.py 2 --amp

# head cross-attention gate lambda
./tools/dist_train.sh configs/ref_yolo_world/ablation/head_scale_context_0.py 2 --amp

# neck cross-attention gate beta
./tools/dist_train.sh configs/ref_yolo_world/ablation/neck_beta_0.py 2 --amp

# both gates fixed to zero
./tools/dist_train.sh configs/ref_yolo_world/ablation/head_lambda0_neck_beta0.py 2 --amp
```
