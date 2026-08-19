"""Shared TNL2K parsing, manifest, and OPE evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np


OVERLAP_THRESHOLDS = np.arange(0.0, 1.0001, 0.05)
PRECISION_THRESHOLDS = np.arange(0.0, 51.0, 1.0)
NORMALIZED_PRECISION_THRESHOLDS = np.arange(0.0, 0.5001, 0.01)
IMAGE_LOAD_ERROR_PREFIX = 'failed to load image:'


@dataclass(frozen=True)
class SequenceData:
    """One TNL2K sequence with aligned frame-level annotations."""

    name: str
    expression: str
    frame_paths: tuple[Path, ...]
    groundtruth: np.ndarray
    absent: np.ndarray


def read_nonempty_text(path: Path) -> str:
    text = path.read_text(encoding='utf-8-sig').strip()
    if not text:
        raise ValueError(f'Empty text file: {path}')
    return ' '.join(text.split())


def read_boxes(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
            path.read_text(encoding='utf-8-sig').splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.replace('\t', ',').split(',')
        if len(fields) != 4:
            fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f'Expected four bbox values at {path}:{line_number}')
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Non-finite bbox at {path}:{line_number}')
        rows.append(values)
    if not rows:
        raise ValueError(f'No bounding boxes in {path}')
    return np.asarray(rows, dtype=np.float64)


def read_absent(path: Path | None, count: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(count, dtype=bool)
    values: list[bool] = []
    for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
        for value in raw_line.replace(',', ' ').split():
            values.append(float(value) != 0.0)
    if len(values) != count:
        raise ValueError(
            f'Absent annotation length mismatch: {path} has {len(values)}, '
            f'expected {count}')
    return np.asarray(values, dtype=bool)


def sorted_frame_paths(frames_dir: Path) -> tuple[Path, ...]:
    paths = sorted(
        path for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {
            '.jpg', '.jpeg', '.png', '.bmp'
        })
    return tuple(paths)


def load_sequence(
        sequence_dir: Path,
        *,
        groundtruth_path: Path | None = None,
        language_path: Path | None = None,
        absent_path: Path | None = None,
        excluded_frame_numbers: Iterable[int] = (),
) -> SequenceData:
    gt_path = groundtruth_path or sequence_dir / 'groundtruth.txt'
    expression_path = language_path or sequence_dir / 'language.txt'
    groundtruth = read_boxes(gt_path)
    frames = list(sorted_frame_paths(sequence_dir / 'frames'))
    excluded = set(excluded_frame_numbers)

    if excluded:
        frame_map = {
            int(path.stem): path
            for path in frames if path.stem.isdigit()
        }
        keep_indices = [
            index for index in range(1, len(groundtruth) + 1)
            if index not in excluded
        ]
        missing = [
            index for index in keep_indices if index not in frame_map
        ]
        if missing:
            raise ValueError(
                f'{sequence_dir.name}: missing frames {missing[:10]}')
        frames = [frame_map[index] for index in keep_indices]
        groundtruth = groundtruth[
            np.asarray([index - 1 for index in keep_indices])]

    if len(frames) != len(groundtruth):
        raise ValueError(
            f'{sequence_dir.name}: {len(frames)} frames but '
            f'{len(groundtruth)} ground-truth rows')

    absent = read_absent(absent_path, len(groundtruth))
    return SequenceData(
        name=sequence_dir.name,
        expression=read_nonempty_text(expression_path),
        frame_paths=tuple(frames),
        groundtruth=groundtruth,
        absent=absent,
    )


def deterministic_split(
        sequence_names: Sequence[str],
        *,
        validation_count: int,
        seed: int,
) -> tuple[list[str], list[str]]:
    names = sorted(sequence_names)
    if validation_count <= 0 or validation_count >= len(names):
        raise ValueError('validation_count must be within the sequence set')
    rng = random.Random(seed)
    rng.shuffle(names)
    validation = sorted(names[:validation_count])
    training = sorted(names[validation_count:])
    return training, validation


def iter_manifest(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'Invalid JSONL at {path}:{line_number}') from error


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def bbox_iou_xywh(prediction: Sequence[float],
                  target: Sequence[float]) -> float:
    pred = xywh_to_xyxy(prediction)
    gt = xywh_to_xyxy(target)
    intersection_width = max(0.0, min(pred[2], gt[2]) - max(pred[0], gt[0]))
    intersection_height = max(0.0,
                              min(pred[3], gt[3]) - max(pred[1], gt[1]))
    intersection = intersection_width * intersection_height
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = pred_area + gt_area - intersection
    return intersection / union if union > 0 else 0.0


def _valid_box(box: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(box)) and box[2] > 0 and box[3] > 0)


def robust_predictions(predictions: np.ndarray,
                       groundtruth: np.ndarray) -> np.ndarray:
    """Apply the official toolkit's invalid-box and first-frame behavior."""
    if predictions.shape != groundtruth.shape or predictions.shape[1] != 4:
        raise ValueError('Prediction and ground-truth arrays must be Nx4')
    result = predictions.astype(np.float64, copy=True)
    for index in range(1, len(result)):
        if not _valid_box(result[index]):
            result[index] = result[index - 1]
    result[0] = groundtruth[0]
    return result


def predict_frames_with_unreadable_fallback(
        frame_paths: Sequence[Path],
        predict_frame: Callable[[Path], np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Preserve frame alignment when the distributed image is unreadable."""
    predictions: list[np.ndarray] = []
    fallbacks: list[dict[str, object]] = []
    for frame_index, frame_path in enumerate(frame_paths, start=1):
        try:
            prediction = np.asarray(
                predict_frame(frame_path), dtype=np.float64)
        except AssertionError as error:
            if not str(error).startswith(IMAGE_LOAD_ERROR_PREFIX):
                raise
            prediction = np.asarray(
                [math.nan, math.nan, 0.0, 0.0], dtype=np.float64)
            fallbacks.append({
                'frame_index': frame_index,
                'frame_path': str(frame_path),
                'error': str(error),
            })
        if prediction.shape != (4,):
            raise ValueError(
                f'Expected one xywh prediction for {frame_path}, '
                f'got shape {prediction.shape}')
        predictions.append(prediction)
    if not predictions:
        raise ValueError('At least one frame is required')
    return np.vstack(predictions), fallbacks


def official_prediction_roundtrip(predictions: np.ndarray) -> np.ndarray:
    """Match the 6-decimal prediction text consumed by the official toolkit."""
    if predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError('Predictions must be an Nx4 array')
    return np.round(predictions.astype(np.float64, copy=False), decimals=6)


def sequence_curves(predictions: np.ndarray, groundtruth: np.ndarray,
                    absent: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Compute toolkit-compatible curves for one sequence."""
    if len(predictions) != len(groundtruth) or len(absent) != len(groundtruth):
        raise ValueError('Frame count mismatch')
    robust = robust_predictions(predictions, groundtruth)
    present = ~absent.astype(bool)
    denominator = len(groundtruth)
    pred_present = robust[present]
    gt_present = groundtruth[present]
    valid_gt = np.all(gt_present > 0, axis=1)
    overlaps = np.full(len(gt_present), -1.0, dtype=np.float64)
    overlaps[valid_gt] = np.asarray([
        bbox_iou_xywh(pred, gt)
        for pred, gt in zip(pred_present[valid_gt], gt_present[valid_gt])
    ])
    raw_overlaps = np.asarray([
        bbox_iou_xywh(pred, gt)
        if _valid_box(pred) else 0.0
        for pred, gt in zip(predictions[present][valid_gt],
                            gt_present[valid_gt])
    ])
    pred_centers = pred_present[:, :2] + (pred_present[:, 2:] - 1.0) / 2.0
    gt_centers = gt_present[:, :2] + (gt_present[:, 2:] - 1.0) / 2.0
    center_errors = np.linalg.norm(pred_centers - gt_centers, axis=1)
    normalized_errors = np.linalg.norm(
        (pred_centers - gt_centers) / np.maximum(gt_present[:, 2:], 1e-12),
        axis=1)
    center_errors[~valid_gt] = -1.0
    normalized_errors[~valid_gt] = -1.0

    success = np.asarray([
        np.count_nonzero(overlaps > threshold) / denominator
        for threshold in OVERLAP_THRESHOLDS
    ])
    precision = np.asarray([
        np.count_nonzero(center_errors <= threshold) / denominator
        for threshold in PRECISION_THRESHOLDS
    ])
    normalized_precision = np.asarray([
        np.count_nonzero(normalized_errors <= threshold) / denominator
        for threshold in NORMALIZED_PRECISION_THRESHOLDS
    ])
    return {
        'success_curve': success,
        'precision_curve': precision,
        'normalized_precision_curve': normalized_precision,
        'frame_acc_at_0_5': float(
            np.mean(raw_overlaps >= 0.5)) if len(raw_overlaps) else 0.0,
        'present_frames': int(np.count_nonzero(valid_gt)),
        'total_frames': denominator,
    }


def aggregate_ope(sequence_results: Sequence[dict]) -> dict:
    if not sequence_results:
        raise ValueError('At least one sequence result is required')
    success = np.mean(
        [result['success_curve'] for result in sequence_results], axis=0)
    precision = np.mean(
        [result['precision_curve'] for result in sequence_results], axis=0)
    normalized = np.mean(
        [result['normalized_precision_curve']
         for result in sequence_results],
        axis=0)
    present_frames = sum(result['present_frames']
                         for result in sequence_results)
    weighted_acc = sum(
        result['frame_acc_at_0_5'] * result['present_frames']
        for result in sequence_results)
    return {
        'success_auc': float(np.mean(success)),
        'precision_at_20px': float(precision[20]),
        'normalized_precision_auc': float(np.mean(normalized)),
        'frame_acc_at_0_5': (
            float(weighted_acc / present_frames) if present_frames else 0.0),
        'sequence_count': len(sequence_results),
        'present_frame_count': present_frames,
        'success_curve': success.tolist(),
        'precision_curve': precision.tolist(),
        'normalized_precision_curve': normalized.tolist(),
    }


def adopt_official_metrics(
        aggregate: dict,
        official_metrics: dict,
        *,
        abs_tol: float = 1e-10,
) -> tuple[dict, dict[str, dict[str, float]]]:
    """Use official metrics as the source of truth and record drift."""
    merged = dict(aggregate)
    mismatches: dict[str, dict[str, float]] = {}
    for key in (
            'success_auc', 'precision_at_20px',
            'normalized_precision_auc'):
        python_value = float(aggregate[key])
        official_value = float(official_metrics[key])
        if not math.isclose(
                python_value, official_value,
                rel_tol=0.0, abs_tol=abs_tol):
            mismatches[key] = {
                'python': python_value,
                'official': official_value,
                'abs_diff': abs(python_value - official_value),
            }
        merged[key] = official_value

    for key in (
            'success_curve', 'precision_curve',
            'normalized_precision_curve'):
        python_curve = np.asarray(aggregate[key], dtype=np.float64)
        official_curve = np.asarray(official_metrics[key], dtype=np.float64)
        if python_curve.shape != official_curve.shape:
            mismatches[key] = {
                'python_length': float(len(python_curve)),
                'official_length': float(len(official_curve)),
            }
        elif not np.allclose(
                python_curve, official_curve,
                rtol=0.0, atol=abs_tol):
            mismatches[key] = {
                'max_abs_diff': float(
                    np.max(np.abs(python_curve - official_curve))),
            }
        merged[key] = official_curve.tolist()
    return merged, mismatches
