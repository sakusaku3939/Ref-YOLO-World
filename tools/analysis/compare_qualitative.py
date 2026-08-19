#!/usr/bin/env python3
"""Compare YOLO-World and Ref-YOLO-World on matched prompts."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Path
    prompt: str
    expression_type: str


def parse_sample(value: str) -> Sample:
    parts = value.split('::', 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            'sample must be SAMPLE_ID::IMAGE::PROMPT::EXPRESSION_TYPE')
    sample_id = parts[0].strip()
    if not re.fullmatch(r'[a-z0-9_-]+', sample_id):
        raise argparse.ArgumentTypeError(
            'sample id must contain only lowercase letters, numbers, _ or -')
    sample = Sample(sample_id, Path(parts[1]), parts[2].strip(), parts[3].strip())
    if not sample.image.is_file():
        raise argparse.ArgumentTypeError(f'image not found: {sample.image}')
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-config', required=True)
    parser.add_argument('--pretrained-checkpoint', required=True)
    parser.add_argument('--ref-config', required=True)
    parser.add_argument('--ref-checkpoint', required=True)
    parser.add_argument('--sample', action='append', type=parse_sample, required=True)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def infer_top1(
        model: torch.nn.Module,
        pipeline: Compose,
        sample: Sample,
        threshold: float) -> dict[str, Any]:
    texts = [[sample.prompt], [' ']]
    model.reparameterize(texts)
    data_info = pipeline(dict(img_id=0, img_path=str(sample.image), texts=texts))
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0),
        data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    predictions = output.pred_instances
    if len(predictions) == 0:
        return {'detected': False, 'score': None, 'bbox': None}
    scores = predictions.scores.float()
    valid = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if len(valid) == 0:
        return {'detected': False, 'score': None, 'bbox': None}
    best = valid[scores[valid].argmax()]
    bbox = predictions.bboxes[best].detach().cpu().tolist()
    return {
        'detected': True,
        'score': float(scores[best].detach().cpu()),
        'bbox': [float(value) for value in bbox],
    }


def run_model(
        config_path: str,
        checkpoint_path: str,
        samples: list[Sample],
        device: str,
        threshold: float) -> list[dict[str, Any]]:
    for path in (config_path, checkpoint_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
    model.eval()
    pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))
    results = [infer_top1(model, pipeline, sample, threshold) for sample in samples]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def annotated_panel(image: np.ndarray, title: str, prediction: dict[str, Any]) -> np.ndarray:
    panel = image.copy()
    if prediction['detected']:
        x1, y1, x2, y2 = (int(round(value)) for value in prediction['bbox'])
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 200, 0), 4)
        score_text = f"score={prediction['score']:.4f}"
    else:
        score_text = 'no detection above threshold'
    header_height = 70
    canvas = np.full((panel.shape[0] + header_height, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_height:] = panel
    cv2.putText(canvas, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(canvas, score_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return canvas


def write_comparisons(
        samples: list[Sample],
        pretrained: list[dict[str, Any]],
        ref_yolo_world: list[dict[str, Any]],
        output_dir: Path) -> list[str]:
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample, pretrained_prediction, ref_prediction in zip(
            samples, pretrained, ref_yolo_world):
        image = cv2.imread(str(sample.image))
        if image is None:
            raise RuntimeError(f'failed to read image: {sample.image}')
        pretrained_panel = annotated_panel(image, 'YOLO-World', pretrained_prediction)
        ref_panel = annotated_panel(image, 'Ref-YOLO-World', ref_prediction)
        comparison = cv2.hconcat([pretrained_panel, ref_panel])
        output = output_dir / f'{sample.sample_id}_comparison.png'
        if not cv2.imwrite(str(output), comparison):
            raise RuntimeError(f'failed to write image: {output}')
        outputs.append(str(output))
    return outputs


def main() -> None:
    args = parse_args()
    samples = args.sample
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pretrained = run_model(
        args.pretrained_config,
        args.pretrained_checkpoint,
        samples,
        args.device,
        args.threshold)
    ref_yolo_world = run_model(
        args.ref_config,
        args.ref_checkpoint,
        samples,
        args.device,
        args.threshold)
    outputs = write_comparisons(
        samples, pretrained, ref_yolo_world, args.output_dir)
    records = []
    for sample, pretrained_prediction, ref_prediction, output in zip(
            samples, pretrained, ref_yolo_world, outputs):
        records.append({
            'sample_id': sample.sample_id,
            'image': str(sample.image),
            'prompt': sample.prompt,
            'expression_type': sample.expression_type,
            'pretrained': pretrained_prediction,
            'ref_yolo_world': ref_prediction,
            'output': output,
        })
    summary = {
        'hypothesis_pass': len(outputs) == len(samples),
        'threshold': args.threshold,
        'topk': 1,
        'pretrained': {
            'config': args.pretrained_config,
            'checkpoint': args.pretrained_checkpoint,
        },
        'ref_yolo_world': {
            'config': args.ref_config,
            'checkpoint': args.ref_checkpoint,
        },
        'device': args.device,
        'n_samples': len(samples),
        'samples': records,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
