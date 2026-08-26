#!/usr/bin/env python3
# Copyright (c) Aishwarya Kamath & Nicolas Carion.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remove evaluation images from RefCOCO-family training annotations.

The image-ID exclusion design is ported from MDETR's RefExp pre-training
scripts. This version reads the JSON list format used by Ref-YOLO-World and
derives the unsafe image IDs directly from all eight published evaluation
splits instead of relying on MDETR's external safe/unsafe ID artifacts.

Upstream source:
https://github.com/ashkamath/mdetr/blob/main/scripts/pre-training/refexp_combined_train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MDETR_SOURCE = (
    'https://github.com/ashkamath/mdetr/blob/main/'
    'scripts/pre-training/refexp_combined_train.py'
)

TRAIN_FILES = {
    'refcoco': Path('refcoco/refcoco/train.json'),
    'refcoco+': Path('refcoco/refcoco+/train.json'),
    'refcocog': Path('refcoco/refcocog/train.json'),
}

EVAL_FILES = {
    'refcoco_val': Path('refcoco/refcoco/validation.json'),
    'refcoco_testA': Path('refcoco/refcoco/testA.json'),
    'refcoco_testB': Path('refcoco/refcoco/testB.json'),
    'refcoco+_val': Path('refcoco/refcoco+/validation.json'),
    'refcoco+_testA': Path('refcoco/refcoco+/testA.json'),
    'refcoco+_testB': Path('refcoco/refcoco+/testB.json'),
    'refcocog_val': Path('refcoco/refcocog/validation.json'),
    'refcocog_test': Path('refcoco/refcocog/test.json'),
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a JSON list')
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'{path}: record {index} is not an object')
        if 'image_id' not in record:
            raise ValueError(f'{path}: record {index} has no image_id')
    return records


def image_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record['image_id']) for record in records}


def build_outputs(
    data_root: Path,
    output_root: Path,
) -> tuple[dict[Path, list[dict[str, Any]]], dict[str, Any]]:
    unsafe_image_ids: set[str] = set()
    evaluation_summary = {}

    for split_name, relative_path in EVAL_FILES.items():
        path = data_root / relative_path
        records = load_records(path)
        split_image_ids = image_ids(records)
        unsafe_image_ids.update(split_image_ids)
        evaluation_summary[split_name] = {
            'path': str(path),
            'samples': len(records),
            'unique_images': len(split_image_ids),
        }

    outputs = {}
    training_summary = {}
    for dataset_name, relative_path in TRAIN_FILES.items():
        source_path = data_root / relative_path
        output_path = output_root / relative_path.relative_to('refcoco')

        records = load_records(source_path)
        raw_image_ids = image_ids(records)
        retained = [
            record for record in records
            if str(record['image_id']) not in unsafe_image_ids
        ]
        retained_image_ids = image_ids(retained)
        removed_image_ids = raw_image_ids & unsafe_image_ids
        remaining_overlap = retained_image_ids & unsafe_image_ids
        if remaining_overlap:
            raise RuntimeError(
                f'{dataset_name}: {len(remaining_overlap)} evaluation images remain')

        outputs[output_path] = retained
        training_summary[dataset_name] = {
            'input': str(source_path),
            'output': str(output_path),
            'raw_samples': len(records),
            'raw_unique_images': len(raw_image_ids),
            'removed_overlap_samples': len(records) - len(retained),
            'removed_overlap_images': len(removed_image_ids),
            'retained_samples': len(retained),
            'retained_unique_images': len(retained_image_ids),
            'remaining_overlap_images': len(remaining_overlap),
        }

    manifest = {
        'provenance': {
            'adapted_from': MDETR_SOURCE,
            'upstream_license': 'Apache-2.0',
            'upstream_copyright': (
                'Copyright (c) Aishwarya Kamath & Nicolas Carion'
            ),
        },
        'policy': 'exclude the union of all eight evaluation splits by image_id',
        'evaluation_unique_images_union': len(unsafe_image_ids),
        'evaluation': evaluation_summary,
        'training': training_summary,
    }
    return outputs, manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Remove images in any RefCOCO-family evaluation split from all '
            'three training annotations.'
        ))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='output root (default: <data-root>/refcoco)',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('results/refcoco_overlap_filter/overlap_manifest.json'),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or args.data_root / 'refcoco'
    try:
        outputs, manifest = build_outputs(args.data_root, output_root)
        for path, records in outputs.items():
            write_json(path, records)
        write_json(args.manifest, manifest)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f'error: {exc}')
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
