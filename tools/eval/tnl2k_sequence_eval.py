#!/usr/bin/env python3
"""Run YOLO-World sequence inference and official-compatible TNL2K OPE."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tnl2k_common import (  # noqa: E402
    aggregate_ope,
    adopt_official_metrics,
    official_prediction_roundtrip,
    predict_frames_with_unreadable_fallback,
    sequence_curves,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument(
        '--data-root', type=Path, default=Path('data/TNL2K_standard'))
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(
            'data/TNL2K_standard/manifests/test_sequences.json'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--limit-sequences', type=int)
    parser.add_argument('--official-toolkit', type=Path)
    parser.add_argument('--octave', default='octave')
    return parser.parse_args()


def _predict_frame(model, pipeline: Compose, image_path: Path,
                   texts: list[list[str]], use_amp: bool) -> np.ndarray:
    data_info = pipeline({
        'img_id': 0,
        'img_path': str(image_path),
        'texts': texts,
    })
    batch = {
        'inputs': data_info['inputs'].unsqueeze(0),
        'data_samples': [data_info['data_samples']],
    }
    # mmcv batched NMS on this stack expects float tensors. Enabling
    # autocast here leaves half-precision boxes/scores in postprocess and
    # crashes before we can materialize the top-1 bbox.
    with autocast(enabled=False), torch.no_grad():
        prediction = model.test_step(batch)[0].pred_instances
    scores = prediction.scores.float()
    eligible = torch.nonzero(scores > 0.0, as_tuple=False).flatten()
    if not len(eligible):
        return np.asarray([math.nan, math.nan, 0.0, 0.0])
    index = eligible[torch.argmax(scores[eligible])]
    x1, y1, x2, y2 = prediction.bboxes[index].detach().cpu().tolist()
    return np.asarray([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)])


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sequences = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.limit_sequences is not None:
        if args.limit_sequences <= 0:
            raise ValueError('--limit-sequences must be positive')
        sequences = sequences[:args.limit_sequences]
    if not sequences:
        raise ValueError('No TNL2K sequences selected')

    model = init_detector(args.config, str(args.checkpoint),
                          device=args.device)
    pipeline = Compose(get_test_pipeline_cfg(model.cfg))
    predictions_dir = args.output_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    sequence_metrics = []
    reparameterize_calls = 0
    unreadable_frame_fallbacks = []

    for sequence in sequences:
        expression = sequence['expression']
        texts = [[expression], [' ']]
        model.reparameterize(texts)
        reparameterize_calls += 1
        relative_paths = [
            Path(relative_path) for relative_path in sequence['frame_paths']
        ]
        predictions, sequence_fallbacks = (
            predict_frames_with_unreadable_fallback(
                relative_paths,
                lambda relative_path: _predict_frame(
                    model,
                    pipeline,
                    args.data_root / relative_path,
                    texts,
                    args.amp,
                ),
            ))
        for fallback in sequence_fallbacks:
            fallback['sequence'] = sequence['sequence']
        unreadable_frame_fallbacks.extend(sequence_fallbacks)
        groundtruth = np.asarray(
            sequence['groundtruth_xywh'], dtype=np.float64)
        absent = np.asarray(sequence['absent'], dtype=bool)
        if len(predictions) != len(groundtruth):
            raise RuntimeError(
                f"{sequence['sequence']}: prediction row mismatch")
        predictions = official_prediction_roundtrip(predictions)
        np.savetxt(
            predictions_dir / f"{sequence['sequence']}.txt",
            predictions,
            fmt='%.6f',
            delimiter=',')
        curves = sequence_curves(predictions, groundtruth, absent)
        sequence_metrics.append({
            'sequence': sequence['sequence'],
            'unreadable_frame_fallback_count': len(sequence_fallbacks),
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in curves.items()
            },
        })

    if reparameterize_calls != len(sequences):
        raise RuntimeError('reparameterize call count mismatch')
    aggregate = aggregate_ope(sequence_metrics)
    python_metrics = dict(aggregate)
    official_metrics = None
    official_metric_mismatch = {}
    if args.official_toolkit is not None:
        if args.limit_sequences is not None:
            raise ValueError(
                'Official evaluation requires the complete 700-sequence set')
        official_path = args.output_dir / 'official_metrics.json'
        command = (
            f"addpath('tools/eval'); tnl2k_official_eval("
            f"'{args.official_toolkit.resolve()}',"
            f"'{predictions_dir.resolve()}',"
            f"'{official_path.resolve()}');")
        subprocess.run(
            [args.octave, '--quiet', '--eval', command],
            check=True)
        official_metrics = json.loads(
            official_path.read_text(encoding='utf-8'))
        aggregate, official_metric_mismatch = adopt_official_metrics(
            aggregate, official_metrics)

    empty_prediction_frames = sum(
        1 for path in predictions_dir.glob('*.txt')
        for line in path.read_text().splitlines()
        if line and ('nan' in line.lower()
                     or line.rstrip().endswith(',0.000000,0.000000')))
    artifact = {
        'config': args.config,
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'manifest': str(args.manifest),
        'sequence_count': len(sequences),
        'reparameterize_calls': reparameterize_calls,
        'score_threshold': 0.0,
        'max_detections': 1,
        'empty_prediction_frames': empty_prediction_frames,
        'unreadable_frame_fallback_policy': (
            'invalid_prediction_then_official_previous_box'),
        'unreadable_frame_fallback_count': len(
            unreadable_frame_fallbacks),
        'unreadable_frame_fallbacks': unreadable_frame_fallbacks,
        'python_metrics': python_metrics,
        'official_toolkit_metrics': official_metrics,
        'official_toolkit_mismatch': official_metric_mismatch,
        'metrics': aggregate,
        'per_sequence': sequence_metrics,
    }
    write_json(args.output_dir / 'metrics.json', artifact)
    print(json.dumps(artifact['metrics'], indent=2))


if __name__ == '__main__':
    main()
