import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'tools' / 'data' / 'filter_refexp_train_overlap.py'
DEFAULT_TRAIN_CONFIG = (
    REPO_ROOT / 'configs' / 'ref_yolo_world'
    / 'ref_yolo_world_multidataset_no_lora.py'
)

TRAIN_PATHS = {
    'refcoco': Path('refcoco/refcoco/train.json'),
    'refcoco+': Path('refcoco/refcoco+/train.json'),
    'refcocog': Path('refcoco/refcocog/train.json'),
}

EVAL_PATHS = (
    Path('refcoco/refcoco/validation.json'),
    Path('refcoco/refcoco/testA.json'),
    Path('refcoco/refcoco/testB.json'),
    Path('refcoco/refcoco+/validation.json'),
    Path('refcoco/refcoco+/testA.json'),
    Path('refcoco/refcoco+/testB.json'),
    Path('refcoco/refcocog/validation.json'),
    Path('refcoco/refcocog/test.json'),
)


def write_records(path: Path, image_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([
            {'image_id': image_id, 'ann_id': index}
            for index, image_id in enumerate(image_ids)
        ]),
        encoding='utf-8',
    )


def test_filters_union_of_all_evaluation_images_and_writes_manifest(tmp_path):
    data_root = tmp_path / 'data'
    manifest_path = tmp_path / 'manifest.json'

    for index, relative_path in enumerate(EVAL_PATHS, start=1):
        write_records(data_root / relative_path, [index, 100])

    write_records(data_root / TRAIN_PATHS['refcoco'], [1, 100, 201])
    write_records(data_root / TRAIN_PATHS['refcoco+'], [4, 100, 202])
    write_records(data_root / TRAIN_PATHS['refcocog'], [8, 203])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--data-root',
            str(data_root),
            '--manifest',
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    expected = {
        'refcoco': [201],
        'refcoco+': [202],
        'refcocog': [203],
    }
    for dataset_name, retained_ids in expected.items():
        output_path = data_root / 'refcoco' / dataset_name / 'train.json'
        records = json.loads(output_path.read_text(encoding='utf-8'))
        assert [record['image_id'] for record in records] == retained_ids

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['evaluation_unique_images_union'] == 9
    assert manifest['provenance']['upstream_license'] == 'Apache-2.0'
    assert 'ashkamath/mdetr' in manifest['provenance']['adapted_from']
    for summary in manifest['training'].values():
        assert summary['remaining_overlap_images'] == 0


def test_rejects_annotation_without_image_id(tmp_path):
    data_root = tmp_path / 'data'
    manifest_path = tmp_path / 'manifest.json'

    for relative_path in EVAL_PATHS:
        write_records(data_root / relative_path, [100])
    for relative_path in TRAIN_PATHS.values():
        write_records(data_root / relative_path, [200])
    invalid_path = data_root / TRAIN_PATHS['refcoco']
    invalid_path.write_text('[{"ann_id": 1}]', encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--data-root',
            str(data_root),
            '--manifest',
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert 'has no image_id' in result.stdout
    assert not manifest_path.exists()


def test_default_joint_training_uses_filtered_annotation_paths():
    config = DEFAULT_TRAIN_CONFIG.read_text(encoding='utf-8')
    for relative_path in TRAIN_PATHS.values():
        assert f"ann_file='{relative_path.as_posix()}'" in config
