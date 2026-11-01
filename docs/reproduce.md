# 🏋️ Training and evaluation

Run every command from the repository root, with the data laid out as described
in [data.md](data.md). We used two NVIDIA L40S GPUs for all reported numbers.

## Training

We trained the released model in two stages.

**Stage 1.** Train every Ref-YOLO-World component except the text LoRA for 20
epochs, starting from the YOLO-World v2-X pretrained weights:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset_no_lora.py \
  2 --amp
```

Copy the best of the 20 checkpoints to
`weights/ref_yolo_world_x_multidataset_no_lora.pth`.

**Stage 2.** Add zero-initialized LoRA adapters to the CLIP text encoder and
train for another 20 epochs from the stage-1 checkpoint. We released the best
checkpoint of this stage, from epoch 10:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  2 --amp
```

The configs below train the same model with fewer components. They form the
module ablation:

| Config | Neck | Spatial PE | Contrastive head | Text LoRA | Train data |
| --- | :-: | :-: | :-: | :-: | --- |
| `configs/yolo_world/finetune_refcocog.py` | – | – | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_t_csp.py` | ✓ | – | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_t_csp_spatial_pe.py` | ✓ | ✓ | – | – | RefCOCOg |
| `configs/ref_yolo_world/ref_yolo_world_refcocog.py` | ✓ | ✓ | ✓ | – | RefCOCOg |
| `configs/ref_yolo_world/ref_yolo_world_multidataset_no_lora.py` | ✓ | ✓ | ✓ | – | RefCOCO/+/g + flip sync |
| `configs/ref_yolo_world/ref_yolo_world_multidataset.py` | ✓ | ✓ | ✓ | ✓ | RefCOCO/+/g + flip sync |

## Accuracy

### All eight RefCOCO / RefCOCO+ / RefCOCOg splits

```bash
python3 tools/eval/refcoco_suite_benchmark.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --gpus 2 \
  --output-dir results/refcoco_suite_accuracy
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

### LVIS minival

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

Stream inference reuses the cached text-side quantities for every image, while
Single inference encodes the expression for every image. Measure end-to-end FPS on
RefCOCOg val with a batch size of one, using
`text_cache_attention_reparameterization_cached.py` for Stream and
`attention_reparameterization_uncached.py` for Single:

```bash
python3 tools/speed/reliable_speed_measure.py \
  --label ref_yolo_world_stream \
  --config configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --measure all --repeat 3 \
  --output results/speed/ref_yolo_world_stream.json
```

Pass another config from the table to measure the other conditions.

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

Attention reparameterization rewrites the cross attention in an algebraically
equivalent form. These two scripts check that it leaves the logits unchanged:

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

The exported models take the cached text-side tensors as inputs and run only
the per-frame detector.

### ONNX

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

### Core ML

Convert the ONNX model to a Core ML ML Program with FP16 compute:

```bash
python3 tools/export/convert_ref_yoloworld_coreml.py \
  --onnx work_dirs/export/ref_yolo_world.onnx \
  --output work_dirs/export/ref_yolo_world_fp16.mlpackage \
  --report work_dirs/export/ref_yolo_world_fp16.coreml.json
```

Add `--image-input` to take an RGB image instead of a normalized tensor, or
`--compute-precision fp32` for FP32 compute. On macOS, the script also checks
the Core ML outputs against ONNX Runtime and writes the errors to the report.

### TFLite

Export a separate ONNX model with `--tflite-strip-cls-score-where`, which works
around an onnx2tf 1.28 shape bug, then convert it and check the outputs against
ONNX Runtime:

```bash
python3 tools/export/export_ref_yoloworld_text_cache_onnx.py \
  configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  weights/ref_yolo_world_x_multidataset.pth \
  --tflite-strip-cls-score-where \
  --output work_dirs/export/ref_yolo_world_tflite.onnx

python3 tools/export/convert_ref_yoloworld_tflite.py \
  --onnx work_dirs/export/ref_yolo_world_tflite.onnx \
  --output-dir work_dirs/export/tflite

python3 tools/export/check_ref_yoloworld_tflite_parity.py \
  --onnx work_dirs/export/ref_yolo_world_tflite.onnx \
  --tflite work_dirs/export/tflite/ref_yolo_world_tflite_float32.tflite
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
