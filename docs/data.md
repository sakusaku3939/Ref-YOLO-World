# Dataset and artifact preparation

All paths below are relative to the repository root.

## Expected layout

```
data/
  coco/train2014/                       COCO 2014 train images (RefCOCO/+/g)
  coco/val2017/                         COCO 2017 validation images (LVIS)
  refcoco/refcoco/{train,validation,testA,testB}.json
  refcoco/refcoco+/{train,validation,testA,testB}.json
  refcoco/refcocog/{train,validation,test}.json
  ref_adv/ref_adv_s.json                Ref-Adv-s (evaluation only)
  TNL2K/                                TNL2K sequences (evaluation only)
  coco/lvis/lvis_v1_minival_inserted_image_name.json
  texts/lvis_v1_class_texts.json
weights/
  yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
  ref_yolo_world_x_multidataset.pth
results/text_cache/
  refcocog_val_text_features.pt
```

## COCO images

RefCOCO, RefCOCO+ and RefCOCOg all annotate COCO 2014 *train* images.

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/train2014.zip -P data/coco
unzip -q data/coco/train2014.zip -d data/coco
```

## RefCOCO / RefCOCO+ / RefCOCOg

The command below fetches every published split from the `jxu124/refcoco`,
`jxu124/refcocoplus`, and `jxu124/refcocog` Hugging Face datasets. It then
removes the union of all eight evaluation splits from the three training
files by image ID. The filtered files are written directly to the paths used
by the default training configs. Run preparation and audit as a mandatory
pre-training gate:

```bash
python3 tools/data/prepare_refcoco_multi_dataset.py --data-root data &&
python3 tools/data/refexp_dataset_audit.py \
  --data-root data \
  --check-images \
  --output results/refexp_audit.json
```

The audit checks all eleven annotation files, bounding boxes, required text,
image files, and image-level train/evaluation separation. Do not start
training unless both commands exit with status zero.

## Ref-Adv-s

```bash
python3 tools/data/prepare_ref_adv_dataset.py --data-root data
```

This downloads 1,142 images and writes `data/ref_adv/ref_adv_s.json` plus a
source manifest. Ref-Adv-s is distributed by
[`dddraxxx/ref-adv-s`](https://huggingface.co/datasets/dddraxxx/ref-adv-s)
under CC-BY-4.0 and is evaluation-only.

## TNL2K

Download the official TNL2K archives, then build the deterministic frame and
sequence manifests:

```bash
python3 tools/eval/prepare_tnl2k_dataset.py
```

## LVIS minival

Formal regression uses the complete 4,809-image LVIS minival annotation.
Download the COCO val2017 images, the GLIP-provided minival annotation used by
YOLO-World, and the standard class text file:

```bash
mkdir -p data/coco/lvis data/texts
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco
unzip -q data/coco/val2017.zip -d data/coco
wget -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json \
  https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
wget -O data/texts/lvis_v1_class_texts.json \
  https://raw.githubusercontent.com/AILab-CVC/YOLO-World/master/data/texts/lvis_v1_class_texts.json
```

Run the complete evaluation with:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh \
  configs/ref_yolo_world/eval/lvis_minival.py \
  weights/ref_yolo_world_x_multidataset.pth \
  2
```

## Weights

Download the pretrained YOLO-World v2-X checkpoint from the upstream
repository and the released Ref-YOLO-World checkpoint from this repository's
GitHub Releases page into `weights/`.

If you start from a plain YOLO-World checkpoint, convert it to the
Ref-YOLO-World parameter layout first:

```bash
python3 tools/convert/convert_yolo_to_ref_checkpoint.py \
  --source weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth \
  --template configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --output weights/ref_yolo_world_x_init.pth
```

## Text feature cache

The cached inference path needs the CLIP text features of every evaluation
expression:

```bash
python3 tools/convert/build_text_feature_cache.py \
  --ann-file data/refcoco/refcocog/validation.json \
  --model-config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --output results/text_cache/refcocog_val_text_features.pt \
  --save-dtype float16
```

For the LoRA-merged evaluation path, merge the adapters into the base linear
layers first and cache the features of the merged encoder:

```bash
python3 tools/convert/merge_text_lora.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \
  --checkpoint weights/ref_yolo_world_x_multidataset.pth \
  --ann-file data/refcoco/refcocog/validation.json \
  --output-checkpoint weights/ref_yolo_world_x_lora_merged.pth \
  --output-json results/lora_merge/merge_report.json
```
