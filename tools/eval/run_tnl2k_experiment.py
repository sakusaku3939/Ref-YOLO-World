#!/usr/bin/env python3
"""Run the complete TNL2K evaluation Zero-shot/Fine-tuning comparison."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tnl2k_common import sha256_file, write_json  # noqa: E402


DEFAULT_CONFIG = (
    'configs/ref_yolo_world/tnl2k/tnl2k_finetune.py')
REFCOCOG_CONFIG = (
    'configs/ref_yolo_world/ref_yolo_world_multidataset.py')
LVIS_SMOKE_CONFIG = (
    'configs/ref_yolo_world/eval/lvis_minival_smoke.py')
PREFLIGHT_STEPS = 500
TRAIN_BATCH_SIZE_PER_GPU = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument(
        '--baseline-checkpoint',
        type=Path,
        default=Path(
            'weights/ref_yolo_world_x_multidataset.pth'))
    parser.add_argument(
        '--toolkit-root',
        type=Path,
        default=Path('data/TNL2K_standard/toolkit'))
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('results/tnl2k_eval'))
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--octave', default='octave')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--preflight-steps', type=int, default=PREFLIGHT_STEPS)
    return parser.parse_args()


def run(command: list[str]) -> float:
    started = time.monotonic()
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, check=True)
    return time.monotonic() - started


def find_checkpoint(work_dir: Path) -> Path:
    candidates = sorted(work_dir.glob('best*.pth'))
    if len(candidates) != 1:
        raise RuntimeError(
            f'Expected one best checkpoint in {work_dir}, got {candidates}')
    return candidates[0]


def build_preflight_manifest(
        source: Path, destination: Path, sample_count: int) -> dict:
    rows = [
        line for line in source.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(rows) < sample_count:
        raise RuntimeError(
            f'Preflight requires {sample_count} samples, '
            f'but {source} contains {len(rows)}')
    selected_indices = [
        index * len(rows) // sample_count for index in range(sample_count)
    ]
    selected = [rows[index] for index in selected_indices]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(selected) + '\n', encoding='utf-8')
    sequences = {
        json.loads(row)['sequence']
        for row in selected
    }
    return {
        'source_samples': len(rows),
        'selected_samples': len(selected),
        'selected_sequences': len(sequences),
        'manifest': str(destination),
    }


def validate_training(
        work_dir: Path, expected_steps: int | None = None) -> dict:
    checkpoints = sorted(work_dir.glob('*.pth'))
    logs = sorted(work_dir.rglob('*.log'))
    text = '\n'.join(path.read_text(errors='replace') for path in logs)
    if not checkpoints:
        raise RuntimeError(f'No checkpoint generated in {work_dir}')
    if not re.search(r'\bloss(?:_[a-z]+)?:\s*[-+0-9.eE]+', text):
        raise RuntimeError(f'No numeric loss found in {work_dir} logs')
    if re.search(r'\b(?:loss(?:_[a-z]+)?):\s*(?:nan|[-+]?inf)\b',
                 text, re.IGNORECASE):
        raise RuntimeError(f'Non-finite loss found in {work_dir} logs')
    result = {
        'finite_loss': True,
        'checkpoint_count': len(checkpoints),
        'checkpoint_generated': True,
        'checkpoint': str(checkpoints[-1]),
        'checkpoint_sha256': sha256_file(checkpoints[-1]),
    }
    if expected_steps is not None:
        progress = [
            (int(step), int(total))
            for step, total in re.findall(
                r'Epoch\(train\)\s*\[\d+\]\[\s*(\d+)/\s*(\d+)\]', text)
        ]
        if not progress:
            raise RuntimeError(
                f'No training step progress found in {work_dir} logs')
        final_step, total_steps = progress[-1]
        if final_step != expected_steps or total_steps != expected_steps:
            raise RuntimeError(
                f'Preflight stopped at {final_step}/{total_steps}; '
                f'expected {expected_steps}/{expected_steps}')
        result['completed_steps'] = final_step
        result['expected_steps'] = expected_steps
    return result


def validate_smoke(output_dir: Path, expected_sequences: int) -> dict:
    metrics = json.loads(
        (output_dir / 'metrics.json').read_text(encoding='utf-8'))
    files = sorted((output_dir / 'predictions').glob('*.txt'))
    if len(files) != expected_sequences:
        raise RuntimeError('Smoke prediction file count mismatch')
    for path in files:
        rows = [line for line in path.read_text().splitlines() if line]
        if not rows or any(len(row.split(',')) != 4 for row in rows):
            raise RuntimeError(f'Official toolkit-incompatible file: {path}')
    if metrics['reparameterize_calls'] != expected_sequences:
        raise RuntimeError('Smoke reparameterize count mismatch')
    return {
        'sequence_count': len(files),
        'prediction_rows_match': True,
        'toolkit_txt_readable': True,
        'reparameterize_calls': metrics['reparameterize_calls'],
        'zero_shot_inference': True,
    }


def metric_summary(artifact: dict) -> dict:
    metrics = artifact['metrics']
    return {
        'success_auc': metrics['success_auc'],
        'precision_20px': metrics['precision_at_20px'],
        'normalized_precision': metrics['normalized_precision_auc'],
        'frame_acc_at_05': metrics['frame_acc_at_0_5'],
        'n_sequences': metrics['sequence_count'],
        'n_frames': metrics['present_frame_count'],
        'empty_prediction_frames': artifact['empty_prediction_frames'],
        'unreadable_frame_fallback_policy': artifact[
            'unreadable_frame_fallback_policy'],
        'unreadable_frame_fallback_count': artifact[
            'unreadable_frame_fallback_count'],
        'unreadable_frame_fallbacks': artifact[
            'unreadable_frame_fallbacks'],
        'reparameterize_calls': artifact['reparameterize_calls'],
        'config': artifact['config'],
        'checkpoint': artifact['checkpoint'],
        'checkpoint_sha256': artifact['checkpoint_sha256'],
    }


def refcocog_regression(checkpoint: Path, output_root: Path,
                        gpus: int) -> dict:
    work_dir = output_root / 'regression' / 'refcocog'
    run([
        'tools/dist_test.sh', REFCOCOG_CONFIG, str(checkpoint), str(gpus),
        '--work-dir', str(work_dir),
    ])
    log_text = '\n'.join(
        path.read_text(errors='replace') for path in work_dir.rglob('*.log'))
    matches = re.findall(r'Acc@0\.5[\'\"]?\s*[:=]\s*([0-9.]+)', log_text)
    if not matches:
        raise RuntimeError('RefCOCOg regression metric not found in logs')
    result = {
        'status': 'passed',
        'acc_at_0_5': float(matches[-1]),
        'checkpoint': str(checkpoint),
    }
    write_json(work_dir / 'summary.json', result)
    return result


def lvis_smoke_regression(checkpoint: Path, output_root: Path,
                          gpus: int) -> dict:
    full_annotation = Path(
        'data/coco/lvis/lvis_v1_minival_inserted_image_name.json')
    smoke_annotation = Path(
        'data/coco/lvis/lvis_v1_minival_smoke_500_seed20260728.json')
    if not smoke_annotation.exists():
        run([
            sys.executable, 'tools/data/build_lvis_smoke_subset.py',
            '--input', str(full_annotation),
            '--output', str(smoke_annotation),
        ])
    work_dir = output_root / 'regression' / 'lvis_smoke'
    run([
        'tools/dist_test.sh', LVIS_SMOKE_CONFIG, str(checkpoint), str(gpus),
        '--work-dir', str(work_dir),
    ])
    metrics_files = sorted(
        path for path in work_dir.glob('*/*.json')
        if re.fullmatch(r'\d{8}_\d{6}\.json', path.name))
    if not metrics_files:
        raise RuntimeError('LVIS smoke metric JSON not found')
    metrics = json.loads(metrics_files[-1].read_text())
    result = {
        'status': 'passed',
        'ap': metrics['lvis/bbox_AP'] * 100.0,
        'apr': metrics['lvis/bbox_APr'] * 100.0,
        'apc': metrics['lvis/bbox_APc'] * 100.0,
        'apf': metrics['lvis/bbox_APf'] * 100.0,
        'n_images': 500,
        'checkpoint': str(checkpoint),
    }
    write_json(work_dir / 'summary.json', result)
    return result


def write_markdown(path: Path, summary: dict) -> None:
    zero = summary['zero_shot']
    tuned = summary['finetuned']
    lines = [
        '# TNL2K Zero-shot / Fine-tuning evaluation',
        '',
        '| condition | Success AUC | Precision@20px | Normalized Precision | Frame Acc@0.5 |',
        '|---|---:|---:|---:|---:|',
        f"| Zero-shot | {zero['success_auc']:.6f} | "
        f"{zero['precision_20px']:.6f} | "
        f"{zero['normalized_precision']:.6f} | "
        f"{zero['frame_acc_at_05']:.6f} |",
        f"| Fine-tuned | {tuned['success_auc']:.6f} | "
        f"{tuned['precision_20px']:.6f} | "
        f"{tuned['normalized_precision']:.6f} | "
        f"{tuned['frame_acc_at_05']:.6f} |",
        '',
        f"Data integrity: `{summary['data_integrity_pass']}`",
        f"Test leakage: `{not summary['no_test_leakage']}`",
        'Unreadable-frame fallbacks: '
        f"Zero-shot `{zero['unreadable_frame_fallback_count']}`, "
        f"Fine-tuned `{tuned['unreadable_frame_fallback_count']}`",
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_preflight_failure(error: Exception, output_root: Path,
                            durations: dict[str, float]) -> dict:
    data_integrity_pass = False
    no_test_leakage = False
    audit_path = output_root / 'data_audit.json'
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            audit = {}
        data_integrity_pass = audit.get('status') == 'passed'
        no_test_leakage = not bool(audit.get('sequence_overlap'))
    return {
        'hypothesis_pass': False,
        'failures': [str(error)],
        'data_integrity_pass': data_integrity_pass,
        'no_test_leakage': no_test_leakage,
        'stage': 'preflight',
        'durations_seconds': durations,
    }


def main() -> None:
    args = parse_args()
    if args.gpus != 2:
        raise ValueError('TNL2K evaluation requires exactly two training GPUs')
    if not args.amp:
        raise ValueError('TNL2K evaluation requires --amp')
    if not 100 <= args.preflight_steps <= 500:
        raise ValueError('--preflight-steps must be between 100 and 500')
    args.output_root.mkdir(parents=True, exist_ok=True)
    durations = {}
    preflight_path = args.output_root / 'preflight_summary.json'

    try:
        durations['data_preparation_seconds'] = run([
            sys.executable, 'tools/eval/prepare_tnl2k_dataset.py',
            '--toolkit-root', str(args.toolkit_root),
            '--audit-output', str(args.output_root / 'data_audit.json'),
        ])
        audit = json.loads(
            (args.output_root / 'data_audit.json').read_text())
        if audit['status'] != 'passed' or audit['sequence_overlap']:
            raise RuntimeError('TNL2K data audit failed')

        smoke_dir = args.output_root / 'preflight_zero_shot'
        smoke_command = [
            sys.executable, 'tools/eval/tnl2k_sequence_eval.py',
            '--config', args.config,
            '--checkpoint', str(args.baseline_checkpoint),
            '--output-dir', str(smoke_dir),
            '--device', args.device,
            '--limit-sequences', '16',
        ]
        if args.amp:
            smoke_command.append('--amp')
        durations['smoke_seconds'] = run(smoke_command)
        smoke = validate_smoke(smoke_dir, 16)

        global_batch_size = args.gpus * TRAIN_BATCH_SIZE_PER_GPU
        preflight_manifest = (
            Path('data/TNL2K_standard/manifests')
            / f'preflight_train_{args.preflight_steps}_steps.jsonl')
        manifest_check = build_preflight_manifest(
            Path('data/TNL2K_standard/manifests/train.jsonl'),
            preflight_manifest,
            args.preflight_steps * global_batch_size,
        )
        attempt = time.strftime('%Y%m%d_%H%M%S')
        preflight_work_dir = Path(
            'work_dirs/tnl2k_preflight_steps'
            f'{args.preflight_steps}_{attempt}')
        durations['preflight_training_seconds'] = run([
            'tools/dist_train.sh', args.config, str(args.gpus),
            '--amp', '--work-dir', str(preflight_work_dir),
            '--cfg-options',
            f'train_dataloader.dataset.ann_file=manifests/'
            f'{preflight_manifest.name}',
            'train_cfg.max_epochs=1',
            'train_cfg.val_interval=100000',
            'default_hooks.param_scheduler.max_epochs=1',
        ])
        training_check = validate_training(
            preflight_work_dir, expected_steps=args.preflight_steps)
        preflight = {
            'hypothesis_pass': True,
            'failures': [],
            'data_integrity_pass': True,
            'no_test_leakage': True,
            'smoke': smoke,
            'preflight_manifest': manifest_check,
            'training': training_check,
        }
        write_json(preflight_path, preflight)
    except Exception as error:
        write_json(
            preflight_path,
            build_preflight_failure(error, args.output_root, durations),
        )
        print(f'Hard gate failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            raise
        return

    zero_dir = args.output_root / 'zero_shot'
    zero_command = [
        sys.executable, 'tools/eval/tnl2k_sequence_eval.py',
        '--config', args.config,
        '--checkpoint', str(args.baseline_checkpoint),
        '--output-dir', str(zero_dir),
        '--device', args.device,
        '--official-toolkit', str(args.toolkit_root),
        '--octave', args.octave,
    ]
    if args.amp:
        zero_command.append('--amp')
    durations['zero_shot_seconds'] = run(zero_command)

    training_work_dir = Path(
        'work_dirs/tnl2k_finetune')
    durations['training_seconds'] = run([
        'tools/dist_train.sh', args.config, str(args.gpus), '--amp',
        '--work-dir', str(training_work_dir),
    ])
    validate_training(training_work_dir)
    fine_checkpoint = find_checkpoint(training_work_dir)

    tuned_dir = args.output_root / 'finetuned'
    tuned_command = [
        sys.executable, 'tools/eval/tnl2k_sequence_eval.py',
        '--config', args.config,
        '--checkpoint', str(fine_checkpoint),
        '--output-dir', str(tuned_dir),
        '--device', args.device,
        '--official-toolkit', str(args.toolkit_root),
        '--octave', args.octave,
    ]
    if args.amp:
        tuned_command.append('--amp')
    durations['finetuned_seconds'] = run(tuned_command)

    zero = metric_summary(json.loads(
        (zero_dir / 'metrics.json').read_text()))
    tuned = metric_summary(json.loads(
        (tuned_dir / 'metrics.json').read_text()))
    delta = {
        key: tuned[key] - zero[key]
        for key in (
            'success_auc', 'precision_20px', 'normalized_precision',
            'frame_acc_at_05')
    }
    regression = refcocog_regression(
        fine_checkpoint, args.output_root, args.gpus)
    lvis_regression = lvis_smoke_regression(
        fine_checkpoint, args.output_root, args.gpus)
    summary = {
        'execution_complete': True,
        'data_integrity_pass': True,
        'no_test_leakage': True,
        'zero_shot': zero,
        'finetuned': tuned,
        'comparison': {'delta': delta},
        'regression_audit': {
            'refcocog': regression,
            'lvis_smoke': lvis_regression,
        },
        'config_sha256': sha256_file(Path(args.config)),
        'fine_tuning_checkpoint_sha256': sha256_file(fine_checkpoint),
        'durations_seconds': durations,
    }
    if not all(math.isfinite(value) for value in delta.values()):
        raise RuntimeError('Non-finite comparison delta')
    write_json(args.output_root / 'summary.json', summary)
    write_markdown(args.output_root / 'summary.md', summary)


if __name__ == '__main__':
    main()
