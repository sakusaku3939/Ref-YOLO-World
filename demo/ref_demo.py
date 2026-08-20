# Copyright (c) Tencent Inc. All rights reserved.
"""
提案手法（Token-Aware Cross-Attention）モデル用の参照表現推論デモ。

引数で画像パスと参照表現を指定して推論し、結果を demo_outputs に保存する。

実行例:
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

# image_demo の推論ロジックを流用
from demo.image_demo import (
    inference_detector,
)
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Token-Aware Cross-Attention モデルで参照表現推論デモ'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ref_yolo_world/ref_yolo_world_multidataset.py',
        help='提案手法の config ファイルパス',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='学習済み checkpoint ファイルパス',
    )
    parser.add_argument(
        '--image',
        type=str,
        default='demo/sample_images/bus.jpg',
        help='入力画像パス（1枚）',
    )
    parser.add_argument(
        '--text',
        type=str,
        default='A man wearing a black coat and sunglasses',
        help='参照表現（1つの自然言語記述）',
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        metavar='K',
        help='最大検出数かつ表示するスコア上位 K 件（image_demo の --topk に合わせる）。例: 1 で1件だけ。',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='信頼度スコアの閾値',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推論デバイス',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='推論結果をウィンドウで表示',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_outputs',
        help='出力画像の保存先ディレクトリ',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='混合精度推論を使用',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='config の上書き（key=value 形式）',
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.isfile(args.image):
        raise FileNotFoundError(f"画像が見つかりません: {args.image}")
    if not osp.isfile(args.checkpoint):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {args.checkpoint}")

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

    # 参照表現1つ + 空白（YOLO-World の仕様）
    texts = [[args.text.strip()], [' ']]
    model.reparameterize(texts)

    os.makedirs(args.output_dir, exist_ok=True)

    parts = [p.strip() for p in args.highlight_color.split(',')]
    highlight_color = tuple(int(p) for p in parts) if len(parts) == 3 else (0, 255, 0)

    # image_demo.inference_detector のシグネチャに合わせてキーワード引数を渡す（互換性維持）
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
    print(f"\n出力保存先: {out_path}")


if __name__ == '__main__':
    main()
