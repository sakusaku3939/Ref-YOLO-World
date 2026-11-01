# Copyright (c) Tencent Inc. All rights reserved.
"""
Referring expression inference demo for the Ref-YOLO-World (Token-Aware Cross-Attention) model.

Runs inference on an image path and a referring expression given as arguments, and saves
the result to demo_outputs.

Example:
  PYTHONPATH=. python demo/ref_demo.py \\
    --config configs/ref_yolo_world/ref_yolo_world_multidataset.py \\
    --checkpoint weights/ref_yolo_world_x_multidataset.pth \\
    --image demo/sample_images/bus.jpg \\
    --text "A man wearing a black coat and sunglasses" \\
    --output-dir demo_outputs
"""

import os
import argparse
import inspect
import os.path as osp

# Reuse the inference logic from image_demo
from demo.image_demo import (
    inference_detector,
)
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Referring expression inference demo with the Token-Aware Cross-Attention model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ref_yolo_world/ref_yolo_world_multidataset.py',
        help='path to the Ref-YOLO-World config file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='path to the trained checkpoint file',
    )
    parser.add_argument(
        '--image',
        type=str,
        default='demo/sample_images/bus.jpg',
        help='path to the input image (a single image)',
    )
    parser.add_argument(
        '--text',
        type=str,
        default='A man wearing a black coat and sunglasses',
        help='referring expression (a single natural-language description)',
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        metavar='K',
        help=('maximum number of detections, i.e. the top-K boxes by score to show '
              '(same as --topk in image_demo). E.g. 1 shows only one box.'),
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='confidence score threshold',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='device used for inference',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='show the inference result in a window',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_outputs',
        help='directory to save output images',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='use mixed precision for inference',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings (key=value format)',
    )
    parser.add_argument(
        '--highlight-top1',
        action='store_true',
        help='draw only the highest-scoring bbox in a different color (default: green).',
    )
    parser.add_argument(
        '--highlight-color',
        type=str,
        default='0,255,0',
        help='color used by --highlight-top1 in BGR "B,G,R" format, e.g. 0,255,0 = green.',
    )
    parser.add_argument(
        '--box-thickness',
        type=int,
        default=1,
        help='bbox line thickness in pixels.',
    )
    parser.add_argument(
        '--highlight-thickness',
        type=int,
        default=3,
        help='line thickness in pixels of the box drawn by --highlight-top1.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not osp.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join(
        './work_dirs',
        osp.splitext(osp.basename(args.config))[0],
    )
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline = Compose(test_pipeline_cfg)

    # One referring expression + a blank entry (required by YOLO-World)
    texts = [[args.text.strip()], [' ']]
    model.reparameterize(texts)

    os.makedirs(args.output_dir, exist_ok=True)

    parts = [p.strip() for p in args.highlight_color.split(',')]
    highlight_color = tuple(int(p) for p in parts) if len(parts) == 3 else (0, 255, 0)

    # Pass only the keyword arguments accepted by image_demo.inference_detector
    sig = inspect.signature(inference_detector)
    valid_params = set(sig.parameters.keys())
    kwargs = dict(
        max_dets=args.topk,
        score_thr=args.threshold,
        output_dir=args.output_dir,
        use_amp=args.amp,
        show=args.show,
        annotation=False,
    )
    if 'highlight_top1' in valid_params:
        kwargs['highlight_top1'] = args.highlight_top1
    if 'highlight_color' in valid_params:
        kwargs['highlight_color'] = highlight_color
    if 'show_topk' in valid_params:
        kwargs['show_topk'] = args.topk
    if 'box_thickness' in valid_params:
        kwargs['box_thickness'] = args.box_thickness
    if 'highlight_thickness' in valid_params:
        kwargs['highlight_thickness'] = args.highlight_thickness

    inference_detector(
        model,
        args.image,
        texts,
        test_pipeline,
        **kwargs,
    )

    out_path = osp.join(args.output_dir, osp.basename(args.image))
    print(f"\nOutput saved to: {out_path}")


if __name__ == '__main__':
    main()
