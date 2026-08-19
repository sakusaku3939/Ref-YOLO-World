#!/usr/bin/env python3
"""Re-measure Ref-YOLO-World (text LoRA) token attention on fixed qualitative samples."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

matplotlib.use('Agg')
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Sample:
    image: Path
    prompt: str
    expression_type: str


def parse_sample(value: str) -> Sample:
    parts = value.split('::', 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            'sample must be IMAGE::PROMPT::EXPRESSION_TYPE')
    sample = Sample(Path(parts[0]), parts[1].strip(), parts[2].strip())
    if not sample.image.is_file():
        raise argparse.ArgumentTypeError(f'image not found: {sample.image}')
    if not sample.prompt or not sample.expression_type:
        raise argparse.ArgumentTypeError('prompt and expression type are required')
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sample', action='append', type=parse_sample, required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=6)
    parser.add_argument('--score-threshold', type=float, default=0.01)
    parser.add_argument(
        '--visualization',
        choices=('heatmap', 'bar', 'both'),
        default='heatmap',
        help='attention visualization to generate (default: heatmap)')
    return parser.parse_args()


def clean_token(token: str) -> str:
    return token.replace('</w>', '').replace('Ġ', '').strip()


def token_labels(model: torch.nn.Module, prompt: str, length: int) -> tuple[list[str], list[int]]:
    tokenizer = model.backbone.text_model.tokenizer
    encoded = tokenizer(text=prompt, return_tensors='pt')
    ids = encoded['input_ids'][0].tolist()[:length]
    labels = [clean_token(tokenizer.convert_ids_to_tokens(token_id)) for token_id in ids]
    special_ids = set(tokenizer.all_special_ids)
    valid = [index for index, token_id in enumerate(ids) if token_id not in special_ids]
    return labels, valid


def top1_prediction(output: Any, score_threshold: float) -> dict[str, Any]:
    predictions = output.pred_instances
    if len(predictions) == 0:
        return {'detected': False, 'score': None, 'bbox': None}
    scores = predictions.scores.float()
    valid = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
    if len(valid) == 0:
        return {'detected': False, 'score': None, 'bbox': None}
    best = valid[scores[valid].argmax()]
    bbox = predictions.bboxes[best].detach().cpu().tolist()
    return {
        'detected': True,
        'score': float(scores[best].detach().cpu()),
        'bbox': [float(value) for value in bbox],
    }


def restore_heatmap(
        heatmap: np.ndarray,
        input_shape: tuple[int, int],
        original_shape: tuple[int, int],
        pad_param: Any) -> np.ndarray:
    input_h, input_w = input_shape
    restored = cv2.resize(heatmap, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    pad = np.asarray(pad_param if pad_param is not None else [0, 0, 0, 0], dtype=float)
    if pad.size != 4:
        raise ValueError(f'unexpected pad_param: {pad_param}')
    top, bottom, left, right = (int(round(value)) for value in pad.tolist())
    y2 = max(top + 1, input_h - bottom)
    x2 = max(left + 1, input_w - right)
    restored = restored[top:y2, left:x2]
    original_h, original_w = original_shape
    return cv2.resize(restored, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def attention_modules(
        model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, str]]:
    selected = []
    for name, module in model.named_modules():
        if hasattr(module, 'guide_value_fc') and hasattr(module, 'guide_fc'):
            module.store_token_attn = True
            selected.append((name, module, 'neck'))

    bbox_head = getattr(model, 'bbox_head', None)
    head_module = getattr(bbox_head, 'head_module', None)
    cls_contrasts = getattr(head_module, 'cls_contrasts', None)
    if cls_contrasts is not None:
        for index, module in enumerate(cls_contrasts):
            if hasattr(module, '_cross_attention'):
                module.store_token_attn = True
                selected.append((f'head_P{index + 3}', module, 'model_head'))
    return selected


def overlay_attention(
        original_rgb: np.ndarray,
        restored: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    low = float(restored.min())
    high = float(restored.max())
    normalized = (restored - low) / max(high - low, 1e-12)
    colored = cv2.applyColorMap(
        np.uint8(np.clip(normalized, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original_rgb + 0.45 * colored)
    stats = {
        'min': low,
        'max': high,
        'mean': float(restored.mean()),
        'std': float(restored.std()),
    }
    return overlay, stats


def make_visualization(
        image_bgr: np.ndarray,
        attention: torch.Tensor,
        labels: list[str],
        valid_indices: list[int],
        input_shape: tuple[int, int],
        pad_param: Any,
        title: str,
        output_path: Path,
        max_tokens: int) -> dict[str, Any]:
    if attention.ndim != 5:
        raise ValueError(f'expected BxHxYxXxL attention, got {tuple(attention.shape)}')
    attention_sample = attention[0].float()
    maps = attention_sample.mean(dim=0).cpu().numpy()
    token_count = maps.shape[-1]
    usable = [index for index in valid_indices if index < token_count]
    if not usable:
        usable = list(range(min(token_count, max_tokens)))
    ranked = sorted(usable, key=lambda index: float(np.std(maps[..., index])), reverse=True)
    selected = sorted(ranked[:max_tokens])
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_shape = image_bgr.shape[:2]

    panel_count = len(selected)
    image_aspect = original_rgb.shape[1] / original_rgb.shape[0]
    panel_width = min(4.0, max(1.2, 3.6 * image_aspect))
    fig, axes = plt.subplots(
        1, panel_count, figsize=(panel_width * panel_count, 4.5))
    axes = np.atleast_1d(axes)
    token_stats = []
    for axis, token_index in zip(axes, selected):
        restored = restore_heatmap(
            maps[..., token_index], input_shape, original_shape, pad_param)
        overlay, stats = overlay_attention(original_rgb, restored)
        label = labels[token_index] if token_index < len(labels) else f'token_{token_index}'
        axis.imshow(overlay)
        axis.set_title(
            f"{label}\nmin={stats['min']:.4f}, max={stats['max']:.4f}")
        axis.axis('off')
        token_stats.append({
            'index': token_index,
            'token': label,
            **stats,
        })
    fig.suptitle(title)
    fig.tight_layout(pad=0.25, w_pad=0.15, rect=(0.0, 0.0, 1.0, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return {
        'output': str(output_path),
        'tokens': token_stats,
        'n_attention_heads': int(attention_sample.shape[0]),
    }


def attention_bar_values(
        attention: torch.Tensor,
        labels: list[str],
        valid_indices: list[int]) -> list[dict[str, Any]]:
    if attention.ndim != 5:
        raise ValueError(f'expected BxHxYxXxL attention, got {tuple(attention.shape)}')
    weights = attention[0].float().mean(dim=(0, 1, 2)).cpu().numpy()
    usable = [index for index in valid_indices if index < len(weights)]
    if not usable:
        usable = list(range(len(weights)))
    return [
        {
            'index': index,
            'token': (
                labels[index]
                if index < len(labels) and labels[index]
                else f'token_{index}'),
            'weight': float(weights[index]),
        }
        for index in usable
    ]


def make_bar_visualization(
        image_bgr: np.ndarray,
        prompt: str,
        prediction: dict[str, Any],
        module_attentions: list[dict[str, Any]],
        output_path: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    axes_flat[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if prediction['detected']:
        x1, y1, x2, y2 = prediction['bbox']
        axes_flat[0].add_patch(plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor='lime',
            linewidth=3))
        prediction_title = f"top-1 score={prediction['score']:.4f}"
    else:
        prediction_title = 'no detection above threshold'
    axes_flat[0].set_title(
        f'{prompt}\n{prediction_title}', fontsize=11, wrap=True)
    axes_flat[0].axis('off')

    module_records = []
    for axis, item in zip(axes_flat[1:], module_attentions):
        tokens = attention_bar_values(
            item['attention'], item['labels'], item['valid_indices'])
        token_names = [
            f"{token['index']}:{token['token']}"
            for token in tokens
        ]
        weights = [token['weight'] for token in tokens]
        axis.barh(token_names, weights, color='steelblue')
        axis.invert_yaxis()
        axis.set_xlabel('Mean attention weight')
        axis.set_title(item['name'].replace('neck.', '').replace('_attn_block', ''))
        axis.grid(axis='x', alpha=0.3)
        module_records.append({
            'module': item['name'],
            'kind': item['kind'],
            'n_attention_heads': int(item['attention'].shape[1]),
            'tokens': tokens,
        })

    for axis in axes_flat[1 + len(module_attentions):]:
        axis.axis('off')
    fig.suptitle(
        'Per-token attention averaged over heads and spatial positions',
        fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return {
        'output': str(output_path),
        'aggregation': 'mean over internal attention heads and HxW spatial positions',
        'modules': module_records,
    }


def run_sample(
        model: torch.nn.Module,
        pipeline: Compose,
        modules: list[tuple[str, torch.nn.Module, str]],
        sample: Sample,
        output_dir: Path,
        max_tokens: int,
        visualization: str,
        score_threshold: float) -> dict[str, Any]:
    for _, module, _ in modules:
        for attr in ('_last_token_attn', '_last_raw_attn'):
            if hasattr(module, attr):
                delattr(module, attr)

    texts = [[sample.prompt], [' ']]
    model.reparameterize(texts)
    data_info = pipeline(dict(img_id=0, img_path=str(sample.image), texts=texts))
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0),
        data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    prediction = top1_prediction(output, score_threshold)

    image_bgr = cv2.imread(str(sample.image))
    if image_bgr is None:
        raise RuntimeError(f'failed to read image: {sample.image}')
    metainfo = data_info['data_samples'].metainfo
    input_shape = tuple(int(value) for value in data_info['inputs'].shape[-2:])
    sample_slug = sample.image.stem
    records = []
    module_attentions = []
    for name, module, kind in modules:
        attention = getattr(module, '_last_token_attn', None)
        if attention is None:
            continue
        labels, valid_indices = token_labels(model, sample.prompt, attention.shape[-1])
        module_attentions.append({
            'name': name,
            'kind': kind,
            'attention': attention,
            'labels': labels,
            'valid_indices': valid_indices,
        })
        record = {
            'module': name,
            'kind': kind,
        }
        if visualization == 'bar':
            records.append(record)
            continue
        module_slug = name.replace('.', '_')
        output_path = output_dir / sample_slug
        if kind == 'model_head':
            output_path /= 'model_heads'
        output_path /= f'{module_slug}.png'
        record.update(make_visualization(
            image_bgr=image_bgr,
            attention=attention,
            labels=labels,
            valid_indices=valid_indices,
            input_shape=input_shape,
            pad_param=metainfo.get('pad_param'),
            title=f'{sample.prompt} | {name}',
            output_path=output_path,
            max_tokens=max_tokens))
        records.append(record)
    if not records:
        raise RuntimeError(f'no token attention was captured for {sample.image}')
    sample_record = {
        'image': str(sample.image),
        'prompt': sample.prompt,
        'expression_type': sample.expression_type,
        'prediction': prediction,
        'visualizations': records,
    }
    if visualization in ('bar', 'both'):
        sample_record['bar_chart'] = make_bar_visualization(
            image_bgr=image_bgr,
            prompt=sample.prompt,
            prediction=prediction,
            module_attentions=module_attentions,
            output_path=output_dir / sample_slug / 'attention_bar_chart.png')
    return sample_record


def main() -> None:
    args = parse_args()
    if not Path(args.config).is_file():
        raise FileNotFoundError(args.config)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    pipeline = Compose(get_test_pipeline_cfg(cfg=cfg))
    modules = attention_modules(model)
    if not modules:
        raise RuntimeError('no token-aware attention modules found')

    samples = [
        run_sample(
            model,
            pipeline,
            modules,
            sample,
            args.output_dir,
            args.max_tokens,
            args.visualization,
            args.score_threshold)
        for sample in args.sample
    ]
    n_neck_captures = sum(
        sum(record['kind'] == 'neck' for record in sample['visualizations'])
        for sample in samples)
    n_model_head_captures = sum(
        sum(record['kind'] == 'model_head' for record in sample['visualizations'])
        for sample in samples)
    n_attention_captures = n_neck_captures + n_model_head_captures
    n_heatmaps = (
        n_attention_captures
        if args.visualization in ('heatmap', 'both')
        else 0)
    n_neck_visualizations = (
        n_neck_captures
        if args.visualization in ('heatmap', 'both')
        else 0)
    n_model_head_visualizations = (
        n_model_head_captures
        if args.visualization in ('heatmap', 'both')
        else 0)
    n_bar_charts = sum('bar_chart' in sample for sample in samples)
    summary = {
        'hypothesis_pass': (
            n_neck_captures > 0
            and n_model_head_captures > 0
            and len(samples) == len(args.sample)),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'visualization': args.visualization,
        'score_threshold': args.score_threshold,
        'aggregation': {
            'heatmap': 'mean over internal attention heads; per-token spatial heatmap',
            'bar': 'mean over internal attention heads and HxW spatial positions',
        },
        'n_samples': len(samples),
        'n_neck_attention_captures': n_neck_captures,
        'n_model_head_attention_captures': n_model_head_captures,
        'n_neck_visualizations': n_neck_visualizations,
        'n_model_head_visualizations': n_model_head_visualizations,
        'n_heatmaps': n_heatmaps,
        'n_bar_charts': n_bar_charts,
        'n_visualizations': n_heatmaps + n_bar_charts,
        'samples': samples,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
