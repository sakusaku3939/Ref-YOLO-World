<div align="center">

# Ref-YOLO-World

**Decoupling Text-Side Computation for Efficient Referring Expression Comprehension on Edge Devices**

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

</div>

**Ref-YOLO-World** is an efficient referring expression comprehension (REC) model
for edge devices. It brings object detection from full sentences such as *"a man
wearing sunglasses and a beige coat"*, not just category names like *"person"*, to
small devices with limited compute such as smartphones. Built on the open-vocabulary
object detector [YOLO-World](https://github.com/AILab-CVC/YOLO-World), it uses
per-token text features while keeping text processing precomputable, so each
expression is encoded only once. It achieves 81.29% Acc@0.5 on RefCOCOg val and
runs at 15.73 FPS on an iPhone 15 Pro.

## Overview

<div align="center">
<img src="assets/overview.png" alt="Overview of Ref-YOLO-World" width="100%">
</div>

Ref-YOLO-World keeps its text branch independent of the image. All text-side
computation, including the key/value projections of the cross-attention, runs
once per expression and is cached, so each frame only runs a one-stage
detector. Per-token text features enter the detection path through two
lightweight modules:

- **Ref-T-CSPLayer** guides each neck stage with all token features through
  cross-attention, together with a 2-D positional encoding for spatial words.
- **Ref-Text Contrastive Head** adds the context of all tokens to each region
  feature before the region–text score.

## Getting started

1. [Installation](docs/installation.md)
2. [Dataset preparation](docs/data.md)
3. [Training and evaluation](docs/reproduce.md)

## Checkpoints

Download the files into `weights/` and `results/text_cache/`; see
[docs/data.md](docs/data.md) for the expected layout.

| File | Description | Download |
| --- | --- | --- |
| `yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth` | YOLO-World v2-X pretrained weights, the starting point of every training run. | [download link](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth) |
| `ref_yolo_world_x_multidataset.pth` | Released Ref-YOLO-World model. | [download link](https://www.dropbox.com/scl/fi/1inbg1m9h68l5q7r8ljf1/ref_yolo_world_x_multidataset.pth?rlkey=s9927hooz33ut9qcwkrsxdafu&dl=1) |
| `refcocog_val_text_features.pt` | Text-feature cache for the cached inference path. | [download link](https://www.dropbox.com/scl/fi/bnsu886ssf4jv911cr8dt/refcocog_val_text_features.pt?rlkey=ip6oc7q9vz99kuet9dr4lbdkv&dl=1) |

## Demo

```bash
python3 demo/ref_demo.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --image demo/sample_images/bus.jpg \
  --text "the person on the left"
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  weights/ref_yolo_world_x_multidataset.pth \
  2
```

Training, the other benchmarks, and the ablations are described in
[docs/reproduce.md](docs/reproduce.md).

## Inference speed

Measure end-to-end FPS on RefCOCOg val with a batch size of one:

```bash
python3 tools/speed/reliable_speed_measure.py \
  --label ref_yolo_world_stream \
  --config configs/ref_yolo_world/speed/text_cache_attention_reparameterization_cached.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --measure all --repeat 3 \
  --output results/speed/ref_yolo_world_stream.json
```

This config measures Stream inference, which reuses the cached text-side quantities.
For Single inference, which encodes the expression for every image, use `configs/ref_yolo_world/speed/attention_reparameterization_uncached.py`.
Other inference-path conditions are listed in [docs/reproduce.md](docs/reproduce.md#inference-speed).

## Acknowledgement

This work builds on [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and
[MMYOLO](https://github.com/open-mmlab/mmyolo), and is released under the same
[GPL-3.0 license](LICENSE).
