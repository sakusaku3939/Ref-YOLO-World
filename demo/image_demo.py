# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import argparse
import os.path as osp
import numpy as np

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--highlight-top1',
        action='store_true',
        help='スコアが最も高い bbox だけ別色で描画する（デフォルト: 緑）。',
    )
    parser.add_argument(
        '--highlight-color',
        type=str,
        default='0,255,0',
        help='highlight-top1 で使う色（BGR の "B,G,R" 形式、例: 0,255,0=緑）。',
    )
    parser.add_argument(
        '--box-thickness',
        type=int,
        default=1,
        help='bbox の線の太さ（ピクセル）。',
    )
    parser.add_argument(
        '--highlight-thickness',
        type=int,
        default=3,
        help='--highlight-top1 で描画する枠の線の太さ（ピクセル）。',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False,
                       highlight_top1=False,
                       highlight_color=(0, 255, 0),
                       show_topk=None,
                       box_thickness=1,
                       highlight_thickness=3):
    data_info = dict(img_id=0, img_path=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    result = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    if show_topk is not None and len(result.xyxy) > show_topk:
        order = np.argsort(result.confidence)[::-1][:show_topk]
        result = sv.Detections(
            xyxy=result.xyxy[order],
            class_id=result.class_id[order],
            confidence=result.confidence[order],
            mask=result.mask[order] if result.mask is not None else None,
        )

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(result.class_id, result.confidence)
    ]

    # label images
    img = cv2.imread(image)
    anno_image = img.copy()
    box_annotator = sv.BoundingBoxAnnotator(thickness=box_thickness)
    img = box_annotator.annotate(img, result)
    if highlight_top1 and len(result.xyxy) > 0:
        top_idx = int(np.argmax(result.confidence))
        x1, y1, x2, y2 = result.xyxy[top_idx]
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            highlight_color,
            thickness=highlight_thickness,
        )
    img = LABEL_ANNOTATOR.annotate(img, result, labels=labels)
    if masks is not None:
        img = MASK_ANNOTATOR.annotate(img, result)
    cv2.imwrite(osp.join(output_dir, osp.basename(image)), img)

    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image)] = anno_image
        annotations_dict[osp.basename(image)] = result

        ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=texts, images=images_dict,
            annotations=annotations_dict).as_yolo(
                annotations_directory_path=ANNOTATIONS_DIRECTORY,
                min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
                max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
                approximation_percentage=APPROXIMATION_PERCENTAGE)

    if show:
        cv2.imshow('Image', img)  # Provide window name
        k = cv2.waitKey(0)
        if k == 27:
            # wait for ESC key to exit
            cv2.destroyAllWindows()

    # print(detections)
    # Detections(xyxy=array([[  5.0775895,  68.32079  , 152.83536  , 325.9108   ],
    #        [  5.0775895,  68.32079  , 152.83536  , 325.9108   ],
    #        [  4.999745 ,  68.408005 , 152.58728  , 325.95456  ],
    #        [  2.627843 ,  15.831497 , 156.02333  , 524.122    ],
    #        [  2.627843 ,  15.831497 , 156.02333  , 524.122    ],
    #        [  2.627843 ,  15.831497 , 156.02333  , 524.122    ],
    #        [  2.3969648,  16.384903 , 156.1233   , 524.0869   ],
    #        [  4.999745 ,  68.408005 , 152.58728  , 325.95456  ],
    #        [  2.3969648,  16.384903 , 156.1233   , 524.0869   ],
    #        [  2.8571224,  15.642172 , 156.13396  , 523.9288   ],
    #        [  4.999745 ,  68.408005 , 152.58728  , 325.95456  ],
    #        [  2.3969648,  16.384903 , 156.1233   , 524.0869   ],
    #        [  5.114694 ,  68.09069  , 152.85773  , 325.74292  ],
    #        [  3.442517 ,  16.3206   , 155.23395  , 524.3288   ],
    #        [  4.999745 ,  68.408005 , 152.58728  , 325.95456  ],
    #        [  2.8571224,  15.642172 , 156.13396  , 523.9288   ],
    #        [  5.114694 ,  68.09069  , 152.85773  , 325.74292  ]],
    #       dtype=float32), mask=None, confidence=array([0.6163826 , 0.36204976, 0.30984783, 0.30524692, 0.26513126,
    #        0.20650044, 0.17271139, 0.16156839, 0.1486191 , 0.1362443 ,
    #        0.13335061, 0.1260023 , 0.1129957 , 0.11005145, 0.07233267,
    #        0.06443565, 0.05152101], dtype=float32), class_id=array([6, 7, 8, 7, 8, 6, 4, 3, 5, 1, 4, 3, 0, 2, 5, 0, 2]), tracker_id=None, data={}, metadata={})

    # 同じラベルの時は、bboxのサイズが大きいものだけを残す
    # result = filter_largest_bbox_per_class_numpy(result)
    result = {
        'xyxy': result.xyxy,
        'class_id': result.class_id,
        'confidence': result.confidence,
    }

    # 各ラベルとスコアをコンソールに出力
    print(f"\n結果（{osp.basename(image)}）:")
    for class_id, confidence in zip(result["class_id"], result["confidence"]):
        label_text = texts[class_id][0]
        print(f"  ラベル: {label_text}, スコア: {confidence:.4f}")


def compute_area_np(bboxes):
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    return widths * heights

def filter_largest_bbox_per_class_numpy(detections):
    xyxy = detections.xyxy
    class_id = detections.class_id
    confidence = detections.confidence

    if xyxy is None or class_id is None or confidence is None:
        return {
            'xyxy': np.empty((0, 4), dtype=np.float32),
            'class_id': np.empty((0,), dtype=np.int64),
            'confidence': np.empty((0,), dtype=np.float32)
        }

    if len(xyxy) == 0:
        return {
            'xyxy': np.empty((0, 4), dtype=np.float32),
            'class_id': np.empty((0,), dtype=np.int64),
            'confidence': np.empty((0,), dtype=np.float32)
        }

    areas = compute_area_np(xyxy)
    unique_classes = np.unique(class_id)

    keep_indices = []

    for cls in unique_classes:
        indices = np.where(class_id == cls)[0]
        if len(indices) > 0:
            cls_areas = areas[indices]
            max_index = indices[np.argmax(cls_areas)]
            keep_indices.append(max_index)

    # index順にフィルタリング
    keep_indices = np.array(keep_indices)

    filtered_detections = {
        'xyxy': xyxy[keep_indices],
        'class_id': class_id[keep_indices],
        'confidence': confidence[keep_indices],
    }

    return filtered_detections



if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    parts = [p.strip() for p in args.highlight_color.split(',')]
    highlight_color = tuple(int(p) for p in parts) if len(parts) == 3 else (0, 255, 0)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    # reparameterize texts
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for image_path in images:
        inference_detector(model,
                           image_path,
                           texts,
                           test_pipeline,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation,
                           highlight_top1=args.highlight_top1,
                           highlight_color=highlight_color,
                           show_topk=args.topk,
                           box_thickness=args.box_thickness,
                           highlight_thickness=args.highlight_thickness)
        progress_bar.update()
