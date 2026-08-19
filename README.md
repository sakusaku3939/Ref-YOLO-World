# Ref-YOLO-World

Real-time Referring Expression Comprehension (REC) built on top of
[YOLO-World](https://github.com/AILab-CVC/YOLO-World).

YOLO-World detects objects from a *category* prompt, but a referring expression
such as *"the left person in a red shirt"* must be resolved against the other
objects in the same image. Ref-YOLO-World keeps the YOLO-World one-stage
detector and adds four components that make it read the expression
token-by-token, plus an inference path that removes the per-query text cost:

| Component | Where | What it does |
| --- | --- | --- |
| **Ref-T-CSPLayer** | Neck | Guides each PAFPN stage with per-token text features instead of a single pooled embedding. |
| **Spatial positional encoding** | Neck | Adds a learnable 2-D positional encoding to the image queries of the cross attention (zero-initialised, so training starts identical to the baseline). |
| **Ref-Text Contrastive Head** | Head | Scores the reference slot with a cross-attention–enriched image feature while keeping the remaining slots on the original contrastive path. |
| **CLIP text LoRA** | Text encoder | Rank-16 zero-init LoRA on `q_proj` / `v_proj` of CLIP text layers 8–11, so the frozen text encoder adapts to referring expressions. |
| **Text feature cache + attention reparameterization** | Inference | Precomputes text features and folds the cross attention into an equivalent linear form, so a cached query skips the text encoder entirely. |

## Results

All numbers below are produced by the configs in this repository on a single
NVIDIA L40S, from the released checkpoint.

Accuracy, expression-complete protocol (every referring expression is one
evaluation sample):

| | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val | test | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Acc@0.5 | 88.09 | 90.81 | 83.59 | 76.77 | 83.50 | 67.66 | 81.62 | 81.52 | **81.70** |

Inference latency on RefCOCOg val, measured with
`tools/speed/reliable_speed_measure.py`:

| Inference path | Latency | Throughput | RefCOCOg val Acc@0.5 |
| --- | ---: | ---: | ---: |
| Standard (text encoder runs per query) | 38.26 ms/image | 26.14 FPS | 80.37 |
| Text cache + attention reparameterization + merged LoRA | **20.08 ms/image** | **49.59 FPS** | 80.37 |

The fast path is an exact rewrite, not an approximation: it reduces latency by
47.5 % while reproducing the accuracy of the standard path to the last correct
sample (2068 / 2573 in both cases; this column uses the older
one-sample-per-target protocol so that the two rows stay comparable with the
speed measurement). `tools/eval/attention_reparameterization_equivalence.py`
and `tools/eval/reparam_cache_equivalence.py` assert the equivalence directly.

## Repository layout

```
configs/
  mmyolo_base/          upstream MMYOLO YOLOv8 base configs (unmodified)
  yolo_world/           upstream YOLO-World v2-X base + RefCOCO/+/g fine-tuning
  ref_yolo_world/       proposed method
    ref_t_csp.py                            Neck only
    ref_t_csp_spatial_pe.py                 + spatial positional encoding
    ref_yolo_world_refcocog.py              + Ref-Text Contrastive Head
    ref_yolo_world_multidataset.py          + RefCOCO/+/g joint training and flip text sync
    ref_yolo_world_multidataset_text_lora.py  + CLIP text LoRA (released model)
    ablation/           module and fine-tuning-scope ablations
    speed/              inference-path ablations (cache / reparameterization / FP32)
    eval/               LVIS minival regression
    tnl2k/              TNL2K sequence-level adaptation
ref_yolo_world/         proposed modules, detector, neck and head
yolo_world/             upstream YOLO-World package (RefCOCO subset)
tools/
  train.py test.py dist_train.sh dist_test.sh
  data/                 dataset preparation and audits
  convert/              checkpoint conversion, LoRA merge, text cache build
  eval/                 accuracy, equivalence and TNL2K evaluation
  speed/                latency measurement and stage breakdown
  export/               ONNX / TFLite export and parity checks
  analysis/             qualitative figures
demo/                   single-image referring expression demo
tests/                  unit tests for the evaluation protocol
```

## Getting started

1. [Installation](docs/installation.md)
2. [Dataset preparation](docs/data.md)
3. [Training and evaluation](docs/reproduce.md)

Quick evaluation of the released checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset_text_lora.py \
  weights/ref_yolo_world_x_multidataset_text_lora.pth \
  2
```

Single-image demo:

```bash
python3 demo/ref_demo.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset_text_lora.py \
  --checkpoint weights/ref_yolo_world_x_multidataset_text_lora.pth \
  --image demo/sample_images/bus.jpg \
  --text "the person on the left"
```

## Checkpoints

Model weights and the precomputed text-feature cache are distributed outside
this repository because of their size. Download them into `weights/` and
`results/text_cache/` respectively; see [docs/data.md](docs/data.md).

| File | Description |
| --- | --- |
| `yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth` | YOLO-World v2-X pretrained weights, the starting point of every training run. |
| `ref_yolo_world_x_multidataset_text_lora.pth` | Released Ref-YOLO-World model (RefCOCOg val 81.62 %). |
| `refcocog_val_text_lora_features.pt` | Text-feature cache for the cached inference path. |

## Acknowledgement

This work builds directly on
[YOLO-World](https://github.com/AILab-CVC/YOLO-World) and
[MMYOLO](https://github.com/open-mmlab/mmyolo). Please cite the original
YOLO-World paper alongside this repository:

```bibtex
@inproceedings{Cheng2024YOLOWorld,
  title     = {YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author    = {Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  booktitle = {CVPR},
  year      = {2024}
}
```

## License

Ref-YOLO-World inherits the GPL-3.0 license of upstream YOLO-World. See
[LICENSE](LICENSE).
