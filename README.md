# Ref-YOLO-World

Real-time Referring Expression Comprehension (REC) built on top of
[YOLO-World](https://github.com/AILab-CVC/YOLO-World).

YOLO-World detects objects from a *category* prompt, but a referring expression
such as *"the left person in a red shirt"* must be resolved against the other
objects in the same image. Ref-YOLO-World keeps the YOLO-World one-stage
detector and adds four components that make it read the expression
token-by-token, plus an inference path that removes the per-query text cost:

## Getting started

1. [Installation](docs/installation.md)
2. [Dataset preparation](docs/data.md)
3. [Training and evaluation](docs/reproduce.md)

Quick evaluation of the released checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  weights/ref_yolo_world_x_multidataset.pth \
  2
```

Single-image demo:

```bash
python3 demo/ref_demo.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --image demo/sample_images/bus.jpg \
  --text "the person on the left"
```

## Checkpoints

Model weights and the precomputed text-feature cache are distributed outside
this repository because of their size. Download them into `weights/` and
`results/text_cache/` respectively; see [docs/data.md](docs/data.md).

| File | Description |
| --- | --- |
| [`yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth`](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/yoloworld_v2_results.md#zero-shot-inference-on-lvis-dataset) | YOLO-World v2-X pretrained weights, the starting point of every training run. |
| `ref_yolo_world_x_multidataset.pth` | Released Ref-YOLO-World model (RefCOCOg val 81.62 %). |
| `refcocog_val_text_features.pt` | Text-feature cache for the cached inference path. |

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
