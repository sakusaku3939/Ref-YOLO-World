#!/usr/bin/env python3
"""Audit TNL2K and create deterministic frame/sequence manifests."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tnl2k_common import (  # noqa: E402
    deterministic_split,
    read_absent,
    read_boxes,
    read_nonempty_text,
    sorted_frame_paths,
    write_json,
)


KNOWN_EXCLUSIONS = {
    'Sunwufan_video_10_done': {
        150: 'source archive contains a corrupt PNG; quarantined separately'
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', type=Path, default=Path('data/TNL2K_standard'))
    parser.add_argument(
        '--toolkit-root',
        type=Path,
        default=Path('data/TNL2K_standard/toolkit'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/TNL2K_standard/manifests'))
    parser.add_argument(
        '--audit-output',
        type=Path,
        default=Path('results/tnl2k_eval/data_audit.json'))
    parser.add_argument('--seed', type=int, default=1152)
    parser.add_argument('--validation-sequences', type=int, default=130)
    return parser.parse_args()


def _find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one {name} below {root}, found {len(matches)}')
    return matches[0]


def _zip_overrides(path: Path | None, filename: str) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            normalized = member.replace('\\', '/')
            if normalized.endswith('/' + filename):
                sequence = normalized.split('/')[-2]
                values[sequence] = archive.read(member).decode(
                    'utf-8-sig').strip()
    return values


def _locate_optional(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous {name}: {matches}')
    return matches[0] if matches else None


def _ensure_official_annotations(toolkit_root: Path) -> None:
    if list(toolkit_root.rglob('gt_rect')) and list(
            toolkit_root.rglob('absent')):
        return
    archive_path = _find_one(toolkit_root, 'annos.tar.gz')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = [
            member for member in archive.getmembers()
            if member.name.startswith(('annos/gt_rect/',
                                       'annos/absent/'))
        ]
        for member in members:
            target = (toolkit_root / member.name).resolve()
            if toolkit_root.resolve() not in target.parents:
                raise ValueError(f'Unsafe archive path: {member.name}')
        archive.extractall(toolkit_root, members=members)


def _parse_boxes_text(text: str, source: str) -> np.ndarray:
    rows = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.strip().replace('\t', ',').split(',')
        if len(fields) != 4:
            fields = raw_line.split()
        if len(fields) != 4:
            raise ValueError(f'Invalid bbox at {source}:{line_number}')
        rows.append([float(value) for value in fields])
    boxes = np.asarray(rows, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f'Invalid annotations in {source}')
    return boxes


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _image_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as handle:
        header = handle.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or len(header) < 24:
        raise ValueError(f'Expected a readable PNG frame: {path}')
    width, height = struct.unpack('>II', header[16:24])
    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:  # pragma: no cover - dataset-dependent
        raise ValueError(f'Unreadable PNG frame: {path}') from error
    return width, height


def _outside_image(bbox: np.ndarray, image_path: Path) -> bool:
    width, height = _image_size(image_path)
    x, y, box_width, box_height = bbox
    return bool(
        x < 0 or y < 0 or x + box_width > width or y + box_height > height)


def _outside_image_or_none(bbox: np.ndarray, image_path: Path) -> bool | None:
    try:
        return _outside_image(bbox, image_path)
    except ValueError:
        return None


def _invalid_training_bbox_reason(bbox: np.ndarray) -> str | None:
    if not np.all(np.isfinite(bbox)):
        return 'non-finite bbox values in source annotation'
    if bbox[2] <= 0 or bbox[3] <= 0:
        return 'non-positive bbox width/height treated as absent frame'
    return None


def _invalid_test_bbox_reason(bbox: np.ndarray, absent: bool) -> str | None:
    del absent
    if not np.all(np.isfinite(bbox)):
        return 'non-finite bbox values in official annotation'
    return None


def _read_absent_lenient(path: Path, count: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(count, dtype=bool)
    try:
        return read_absent(path, count)
    except ValueError:
        pass
    raw_values: list[bool] = []
    for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
        for value in raw_line.replace(',', ' ').split():
            raw_values.append(float(value) != 0.0)
    if len(raw_values) >= count:
        return np.asarray(raw_values[:count], dtype=bool)
    padded = raw_values + [False] * (count - len(raw_values))
    return np.asarray(padded, dtype=bool)


def _training_sequences(data_root: Path, toolkit_root: Path) -> tuple[
        dict[str, dict], list[dict], int, int, int]:
    sequence_root = data_root / 'train' / 'TNL2K_train'
    if not sequence_root.is_dir():
        raise FileNotFoundError(sequence_root)

    revised = _zip_overrides(
        _locate_optional(toolkit_root, 'revised_annotations.zip'),
        'groundtruth.txt')
    updated_archive = _locate_optional(
        toolkit_root, 'updated_4_video_text_annotation.zip')
    updated_groundtruth = _zip_overrides(
        updated_archive, 'groundtruth.txt')
    text_updates = _zip_overrides(
        updated_archive,
        'language.txt')

    sequences = {}
    exclusions = []
    outside_image_count = 0
    invalid_bbox_count = 0
    unreadable_frame_count = 0
    for sequence_dir in sorted(path for path in sequence_root.iterdir()
                               if path.is_dir()):
        name = sequence_dir.name
        if name in updated_groundtruth:
            boxes = _parse_boxes_text(
                updated_groundtruth[name], f'official_updated:{name}')
            annotation_source = 'official_updated'
        elif name in revised:
            boxes = _parse_boxes_text(revised[name], f'revised:{name}')
            annotation_source = 'official_revised'
        else:
            boxes = read_boxes(sequence_dir / 'groundtruth.txt')
            annotation_source = 'distribution'
        expression = (
            ' '.join(text_updates[name].split()) if name in text_updates
            else read_nonempty_text(sequence_dir / 'language.txt')
        )
        frames = sorted_frame_paths(sequence_dir / 'frames')
        frame_map = {
            int(path.stem): path for path in frames if path.stem.isdigit()
        }
        excluded = KNOWN_EXCLUSIONS.get(name, {})
        rows = []
        for frame_number, bbox in enumerate(boxes, start=1):
            if frame_number in excluded:
                exclusions.append({
                    'sequence': name,
                    'frame_index': frame_number,
                    'reason': excluded[frame_number],
                })
                continue
            frame_path = frame_map.get(frame_number)
            if frame_path is None:
                raise ValueError(f'{name}: missing frame {frame_number}')
            invalid_reason = _invalid_training_bbox_reason(bbox)
            if invalid_reason is not None:
                invalid_bbox_count += 1
                exclusions.append({
                    'sequence': name,
                    'frame_index': frame_number,
                    'reason': invalid_reason,
                })
                continue
            outside_image = _outside_image_or_none(bbox, frame_path)
            if outside_image is None:
                unreadable_frame_count += 1
                exclusions.append({
                    'sequence': name,
                    'frame_index': frame_number,
                    'reason': 'unreadable PNG frame excluded from training manifest',
                })
                continue
            else:
                outside_image_count += int(outside_image)
            rows.append({
                'frame_index': frame_number,
                'frame_path': frame_path,
                'bbox_xywh': bbox.tolist(),
            })
        unexpected = sorted(set(frame_map) - set(range(1, len(boxes) + 1)))
        if unexpected:
            raise ValueError(f'{name}: unexpected frames {unexpected[:10]}')
        sequences[name] = {
            'expression': expression,
            'rows': rows,
            'annotation_source': annotation_source,
            'text_source': (
                'official_updated' if name in text_updates else 'distribution'),
        }
    return (
        sequences,
        exclusions,
        outside_image_count,
        invalid_bbox_count,
        unreadable_frame_count,
    )


def _testing_sequences(data_root: Path,
                       toolkit_root: Path,
                       expected_sequence_count: int | None = None) -> tuple[
                           dict[str, dict], int, int, list[str], list[str]]:
    sequence_root = data_root / 'test' / 'video_seqs'
    list_path = _find_one(toolkit_root, 'TNL2K_testing_set.txt')
    names = [
        line.strip() for line in list_path.read_text(
            encoding='utf-8-sig').splitlines() if line.strip()
    ]
    gt_root = _find_one(toolkit_root, 'gt_rect')
    absent_root = _find_one(toolkit_root, 'absent')
    if not gt_root.is_dir() or not absent_root.is_dir():
        raise NotADirectoryError('Official annotations must be extracted')
    updated_archive = _locate_optional(
        toolkit_root, 'updated_4_video_text_annotation.zip')
    updated_groundtruth = _zip_overrides(
        updated_archive, 'groundtruth.txt')
    updated_text = _zip_overrides(updated_archive, 'language.txt')

    sequences = {}
    outside_image_count = 0
    unreadable_frame_count = 0
    for name in names:
        sequence_dir = sequence_root / name
        frames = sorted_frame_paths(sequence_dir / 'frames')
        if name in updated_groundtruth:
            updated_path = gt_root / f'{name}.txt'
            updated_path.write_text(
                updated_groundtruth[name].strip() + '\n', encoding='utf-8')
        boxes = read_boxes(gt_root / f'{name}.txt')
        absent = _read_absent_lenient(absent_root / f'{name}.txt', len(boxes))
        if len(frames) != len(boxes):
            raise ValueError(
                f'{name}: {len(frames)} test frames but {len(boxes)} GT rows')
        for frame_path, bbox, frame_absent in zip(frames, boxes, absent):
            invalid_reason = _invalid_test_bbox_reason(
                bbox, bool(frame_absent))
            if invalid_reason is not None:
                raise ValueError(
                    f'{name}: invalid official test bbox')
            if not frame_absent:
                outside_image = _outside_image_or_none(bbox, frame_path)
                if outside_image is None:
                    unreadable_frame_count += 1
                else:
                    outside_image_count += int(outside_image)
        sequences[name] = {
            'expression': (
                ' '.join(updated_text[name].split())
                if name in updated_text else
                read_nonempty_text(sequence_dir / 'language.txt')),
            'frames': frames,
            'groundtruth': boxes,
            'absent': absent,
        }
    if (expected_sequence_count is not None
            and len(sequences) != expected_sequence_count):
        raise ValueError(
            f'Expected {expected_sequence_count} test sequences, '
            f'got {len(sequences)}')
    return (
        sequences,
        outside_image_count,
        unreadable_frame_count,
        sorted(set(names) & set(updated_groundtruth)),
        sorted(set(names) & set(updated_text)),
    )


def main() -> None:
    args = parse_args()
    _ensure_official_annotations(args.toolkit_root)
    (training, exclusions, train_outside_count, invalid_bbox_count,
     train_unreadable_count) = _training_sequences(
        args.data_root, args.toolkit_root)
    (testing, test_outside_count, test_unreadable_count, updated_test_groundtruth,
     updated_test_text) = _testing_sequences(
         args.data_root, args.toolkit_root, expected_sequence_count=700)
    train_test_overlap = sorted(set(training) & set(testing))
    for name in train_test_overlap:
        training.pop(name, None)
        exclusions.append({
            'sequence': name,
            'frame_index': None,
            'reason': 'removed entire training sequence due to official test name overlap',
        })
    train_names, validation_names = deterministic_split(
        list(training),
        validation_count=args.validation_sequences,
        seed=args.seed)
    if set(train_names) & set(validation_names):
        raise RuntimeError('Training and validation sequence leakage')

    next_image_id = 0
    split_rows = {'train': [], 'validation': []}
    for split, names in (
            ('train', train_names), ('validation', validation_names)):
        for name in names:
            sequence = training[name]
            for frame in sequence['rows']:
                split_rows[split].append({
                    'img_id': next_image_id,
                    'img_path': str(
                        frame['frame_path'].relative_to(args.data_root)),
                    'bbox_xywh': frame['bbox_xywh'],
                    'expression': sequence['expression'],
                    'sequence': name,
                    'frame_index': frame['frame_index'],
                    'split': split,
                })
                next_image_id += 1
        _write_manifest(
            args.output_dir / f'{split}.jsonl', split_rows[split])

    test_manifest = []
    for name, sequence in testing.items():
        test_manifest.append({
            'sequence': name,
            'expression': sequence['expression'],
            'frame_paths': [
                str(path.relative_to(args.data_root))
                for path in sequence['frames']
            ],
            'groundtruth_xywh': sequence['groundtruth'].tolist(),
            'absent': sequence['absent'].astype(int).tolist(),
        })
    write_json(args.output_dir / 'test_sequences.json', test_manifest)
    write_json(args.output_dir / 'split.json', {
        'seed': args.seed,
        'training_sequences': train_names,
        'validation_sequences': validation_names,
    })

    audit = {
        'status': 'passed',
        'data_root': str(args.data_root.resolve()),
        'toolkit_root': str(args.toolkit_root.resolve()),
        'split_seed': args.seed,
        'sequence_counts': {
            'training': len(train_names),
            'validation': len(validation_names),
            'testing': len(testing),
        },
        'frame_counts': {
            'training': len(split_rows['train']),
            'validation': len(split_rows['validation']),
            'testing': sum(len(item['frames']) for item in testing.values()),
        },
        'sequence_overlap': [],
        'removed_sequence_overlap': train_test_overlap,
        'bbox_row_counts': {
            'training': len(split_rows['train']),
            'validation': len(split_rows['validation']),
            'testing': sum(
                len(item['groundtruth']) for item in testing.values()),
        },
        'expression_counts': {
            'training': len(train_names),
            'validation': len(validation_names),
            'testing': len(testing),
        },
        'invalid_bbox_count': invalid_bbox_count,
        'out_of_image_bbox_count': {
            'training_and_validation': train_outside_count,
            'testing': test_outside_count,
        },
        'unreadable_frame_count': {
            'training_and_validation': train_unreadable_count,
            'testing': test_unreadable_count,
        },
        'excluded_frames': exclusions,
        'revised_annotation_sequences': sorted(
            name for name, item in training.items()
            if item['annotation_source'] == 'official_revised'),
        'updated_groundtruth_sequences': sorted([
            name for name, item in training.items()
            if item['annotation_source'] == 'official_updated'
        ] + updated_test_groundtruth),
        'updated_text_sequences': sorted(
            [name for name, item in training.items()
             if item['text_source'] == 'official_updated']
            + updated_test_text),
    }
    write_json(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
