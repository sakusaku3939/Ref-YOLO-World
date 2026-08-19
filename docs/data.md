# Dataset and artifact preparation

All paths below are relative to the repository root.

## Expected layout

```
data/
  coco/train2014/                       COCO 2014 train images (RefCOCO/+/g)
  refcoco/refcoco/{train,validation,testA,testB}.json
  refcoco/refcoco+/{train,validation,testA,testB}.json
  refcoco/refcocog/{train,validation,test}.json
  ref_adv/ref_adv_s.json                Ref-Adv-s (evaluation only)
  TNL2K/                                TNL2K sequences (evaluation only)
  coco/lvis/lvis_v1_minival_inserted_image_name.json
weights/
  yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
  ref_yolo_world_x_multidataset_text_lora.pth
results/text_cache/
  refcocog_val_text_lora_features.pt
```

## COCO images

RefCOCO, RefCOCO+ and RefCOCOg all annotate COCO 2014 *train* images.

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/train2014.zip -P data/coco
unzip -q data/coco/train2014.zip -d data/coco
```

## RefCOCO / RefCOCO+ / RefCOCOg

The annotation JSONs are fetched from Hugging Face and written in the shape
consumed by `CustomRefCOCODataset`:

```bash
python3 tools/data/prepare_refcoco_multi_dataset.py --data-root data
```

Audit the resulting splits (counts, box sanity, train/val overlap) with:

```bash
python3 tools/data/refexp_dataset_audit.py \
  --ann-file data/refcoco/refcocog/validation.json \
  --output results/refexp_audit.json
```

## Ref-Adv-s (adversarial evaluation set)

```bash
python3 tools/data/prepare_ref_adv_dataset.py --data-root data
```

Ref-Adv-s is evaluation-only. It must never be used for training, early
stopping or threshold selection.

## TNL2K (sequence-level evaluation)

Download the official TNL2K archives, then build the deterministic frame and
sequence manifests:

```bash
python3 tools/eval/prepare_tnl2k_dataset.py
```

## LVIS minival (open-vocabulary regression check)

Used to confirm that RefCOCO fine-tuning does not destroy the open-vocabulary
ability inherited from YOLO-World.

```bash
python3 tools/data/build_lvis_smoke_subset.py
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
  --template configs/ref_yolo_world/ref_yolo_world_multidataset_text_lora.py \
  --output weights/ref_yolo_world_x_init.pth
```

## Text feature cache

The cached inference path needs the CLIP text features of every evaluation
expression:

```bash
python3 tools/convert/build_text_feature_cache.py \
  --ann-file data/refcoco/refcocog/validation.json \
  --model-config configs/ref_yolo_world/ref_yolo_world_multidataset_text_lora.py \
  --checkpoint weights/ref_yolo_world_x_multidataset_text_lora.pth \
  --output results/text_cache/refcocog_val_text_lora_features.pt \
  --save-dtype float16
```

For the LoRA-merged evaluation path, merge the adapters into the base linear
layers first and cache the features of the merged encoder:

```bash
python3 tools/convert/merge_text_lora.py \
  --config configs/ref_yolo_world/ref_yolo_world_multidataset_text_lora.py \
  --checkpoint weights/ref_yolo_world_x_multidataset_text_lora.pth \
  --ann-file data/refcoco/refcocog/validation.json \
  --output-checkpoint weights/ref_yolo_world_x_lora_merged.pth \
  --output-json results/lora_merge/merge_report.json
```
