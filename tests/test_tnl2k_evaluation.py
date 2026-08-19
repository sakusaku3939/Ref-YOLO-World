import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'tools' / 'eval'))

from prepare_tnl2k_dataset import (  # noqa: E402
    _read_absent_lenient,
    _testing_sequences,
    _training_sequences,
)
from run_tnl2k_experiment import (  # noqa: E402
    build_preflight_manifest,
    validate_training,
)
from tnl2k_common import (  # noqa: E402
    aggregate_ope,
    adopt_official_metrics,
    bbox_iou_xywh,
    deterministic_split,
    official_prediction_roundtrip,
    predict_frames_with_unreadable_fallback,
    robust_predictions,
    sequence_curves,
)


def _write_valid_png(path: Path) -> None:
    Image.new('RGB', (16, 16)).save(path)


def test_deterministic_split_is_sequence_level_and_reproducible():
    names = [f'sequence_{index:04d}' for index in range(1300)]
    first_train, first_validation = deterministic_split(
        names, validation_count=130, seed=1152)
    second_train, second_validation = deterministic_split(
        list(reversed(names)), validation_count=130, seed=1152)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 1170
    assert len(first_validation) == 130
    assert not set(first_train) & set(first_validation)


def test_bbox_iou_xywh():
    assert bbox_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou_xywh([0, 0, 10, 10], [10, 10, 10, 10]) == 0.0
    assert np.isclose(
        bbox_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3)


def test_official_first_frame_and_failure_box_behavior():
    groundtruth = np.asarray([
        [10, 10, 10, 10],
        [11, 10, 10, 10],
        [12, 10, 10, 10],
    ], dtype=float)
    predictions = np.asarray([
        [100, 100, 10, 10],
        [np.nan, np.nan, 0, 0],
        [12, 10, 10, 10],
    ])
    robust = robust_predictions(predictions, groundtruth)
    np.testing.assert_array_equal(robust[0], groundtruth[0])
    np.testing.assert_array_equal(robust[1], predictions[0])
    np.testing.assert_array_equal(robust[2], predictions[2])


def test_unreadable_frame_fallback_preserves_alignment_and_records_path():
    frame_paths = [Path('demo/00001.png'), Path('demo/00002.png')]

    def predict_frame(path: Path) -> np.ndarray:
        if path.name == '00002.png':
            raise AssertionError(f'failed to load image: {path}')
        return np.asarray([10.0, 20.0, 30.0, 40.0])

    predictions, fallbacks = predict_frames_with_unreadable_fallback(
        frame_paths, predict_frame)

    assert predictions.shape == (2, 4)
    np.testing.assert_array_equal(
        predictions[0], np.asarray([10.0, 20.0, 30.0, 40.0]))
    assert np.isnan(predictions[1, :2]).all()
    np.testing.assert_array_equal(predictions[1, 2:], np.zeros(2))
    np.testing.assert_array_equal(
        robust_predictions(
            predictions,
            np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float),
        )[1],
        predictions[0],
    )
    assert fallbacks == [{
        'frame_index': 2,
        'frame_path': 'demo/00002.png',
        'error': 'failed to load image: demo/00002.png',
    }]


def test_unreadable_frame_fallback_does_not_hide_other_assertions():
    def predict_frame(_path: Path) -> np.ndarray:
        raise AssertionError('model output invariant failed')

    with pytest.raises(AssertionError, match='model output invariant failed'):
        predict_frames_with_unreadable_fallback(
            [Path('demo/00001.png')], predict_frame)


def test_ope_curves_use_official_strict_overlap_and_original_denominator():
    groundtruth = np.asarray([
        [1, 1, 10, 10],
        [1, 1, 10, 10],
    ], dtype=float)
    predictions = groundtruth.copy()
    result = sequence_curves(
        predictions, groundtruth, np.asarray([False, True]))

    # One present frame over the official two-frame denominator.
    assert result['success_curve'][0] == 0.5
    assert result['success_curve'][-1] == 0.0
    assert result['precision_curve'][20] == 0.5
    assert result['normalized_precision_curve'][0] == 0.5
    assert result['frame_acc_at_0_5'] == 1.0


def test_aggregate_ope_uses_sequence_average_and_frame_weighted_acc():
    first = {
        'success_curve': np.ones(21),
        'precision_curve': np.ones(51),
        'normalized_precision_curve': np.ones(51),
        'frame_acc_at_0_5': 1.0,
        'present_frames': 1,
        'total_frames': 1,
    }
    second = {
        'success_curve': np.zeros(21),
        'precision_curve': np.zeros(51),
        'normalized_precision_curve': np.zeros(51),
        'frame_acc_at_0_5': 0.0,
        'present_frames': 3,
        'total_frames': 3,
    }
    aggregate = aggregate_ope([first, second])
    assert aggregate['success_auc'] == 0.5
    assert aggregate['precision_at_20px'] == 0.5
    assert aggregate['normalized_precision_auc'] == 0.5
    assert aggregate['frame_acc_at_0_5'] == 0.25


def test_adopt_official_metrics_prefers_official_and_records_drift():
    aggregate = {
        'success_auc': 0.1,
        'precision_at_20px': 0.2,
        'normalized_precision_auc': 0.3,
        'frame_acc_at_0_5': 0.4,
        'sequence_count': 2,
        'present_frame_count': 5,
        'success_curve': [0.1, 0.0],
        'precision_curve': [0.2, 0.3],
        'normalized_precision_curve': [0.4, 0.5],
    }
    official = {
        'success_auc': 0.11,
        'precision_at_20px': 0.2,
        'normalized_precision_auc': 0.3,
        'success_curve': [0.11, 0.01],
        'precision_curve': [0.2, 0.3],
        'normalized_precision_curve': [0.4, 0.5],
    }

    merged, mismatches = adopt_official_metrics(aggregate, official)

    assert merged['success_auc'] == official['success_auc']
    assert merged['success_curve'] == official['success_curve']
    assert merged['frame_acc_at_0_5'] == aggregate['frame_acc_at_0_5']
    assert mismatches['success_auc']['python'] == aggregate['success_auc']
    assert mismatches['success_auc']['official'] == official['success_auc']
    assert mismatches['success_curve']['max_abs_diff'] == 0.01


def test_official_prediction_roundtrip_matches_saved_txt_precision():
    predictions = np.asarray([
        [1.12345649, 2.98765451, 10.00000049, 20.00000051],
        [np.nan, np.nan, 0.0, 0.0],
    ])

    rounded = official_prediction_roundtrip(predictions)

    np.testing.assert_allclose(
        rounded[0], np.asarray([1.123456, 2.987655, 10.0, 20.000001]))
    assert np.isnan(rounded[1, :2]).all()
    np.testing.assert_array_equal(rounded[1, 2:], np.zeros(2))


def test_training_manifest_excludes_invalid_bbox_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_root = root / 'data'
        toolkit_root = root / 'toolkit'
        sequence_dir = data_root / 'train' / 'TNL2K_train' / 'demo'
        frames_dir = sequence_dir / 'frames'
        frames_dir.mkdir(parents=True)
        toolkit_root.mkdir(parents=True)

        for frame_index in range(1, 4):
            _write_valid_png(frames_dir / f'{frame_index:05d}.png')
        (sequence_dir / 'groundtruth.txt').write_text(
            '1,2,3,4\n0,0,0,0\n5,6,7,8\n', encoding='utf-8')
        (sequence_dir / 'language.txt').write_text('demo target\n',
                                                   encoding='utf-8')

        (sequences, exclusions, outside_count, invalid_count,
         unreadable_count) = _training_sequences(
            data_root, toolkit_root)

    assert list(sequences) == ['demo']
    assert [row['frame_index'] for row in sequences['demo']['rows']] == [1, 3]
    assert invalid_count == 1
    assert outside_count == 0
    assert unreadable_count == 0
    assert exclusions == [{
        'sequence': 'demo',
        'frame_index': 2,
        'reason': 'non-positive bbox width/height treated as absent frame',
    }]


def test_training_manifest_excludes_unreadable_png_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_root = root / 'data'
        toolkit_root = root / 'toolkit'
        sequence_dir = data_root / 'train' / 'TNL2K_train' / 'demo'
        frames_dir = sequence_dir / 'frames'
        frames_dir.mkdir(parents=True)
        toolkit_root.mkdir(parents=True)

        _write_valid_png(frames_dir / '00001.png')
        (frames_dir / '00002.png').write_bytes(b'not a PNG')
        (sequence_dir / 'groundtruth.txt').write_text(
            '1,2,3,4\n5,6,7,8\n', encoding='utf-8')
        (sequence_dir / 'language.txt').write_text(
            'demo target\n', encoding='utf-8')

        (sequences, exclusions, outside_count, invalid_count,
         unreadable_count) = _training_sequences(data_root, toolkit_root)

    assert [row['frame_index'] for row in sequences['demo']['rows']] == [1]
    assert outside_count == 0
    assert invalid_count == 0
    assert unreadable_count == 1
    assert exclusions == [{
        'sequence': 'demo',
        'frame_index': 2,
        'reason': 'unreadable PNG frame excluded from training manifest',
    }]


def test_preflight_manifest_is_deterministic_and_spans_sequences():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / 'train.jsonl'
        first = root / 'first.jsonl'
        second = root / 'second.jsonl'
        rows = [
            {
                'img_id': index,
                'sequence': f'sequence_{index // 10}',
            }
            for index in range(100)
        ]
        source.write_text(
            ''.join(f'{json.dumps(row)}\n' for row in rows),
            encoding='utf-8')

        first_result = build_preflight_manifest(source, first, 20)
        second_result = build_preflight_manifest(source, second, 20)
        first_text = first.read_text()
        second_text = second.read_text()

    assert first_text == second_text
    assert first_result == {
        'source_samples': 100,
        'selected_samples': 20,
        'selected_sequences': 10,
        'manifest': str(first),
    }
    assert second_result['manifest'] == str(second)


def test_validate_training_requires_completed_preflight_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        (work_dir / 'epoch_1.pth').write_bytes(b'checkpoint')
        (work_dir / 'train.log').write_text(
            'Epoch(train) [1][  450/500] loss: 2.0\n'
            'Epoch(train) [1][  500/500] loss: 1.0\n',
            encoding='utf-8')

        result = validate_training(work_dir, expected_steps=500)

    assert result['finite_loss'] is True
    assert result['checkpoint_generated'] is True
    assert result['completed_steps'] == 500
    assert result['expected_steps'] == 500


def test_testing_sequences_allow_absent_zero_bbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_root = root / 'data'
        toolkit_root = root / 'toolkit'
        sequence_dir = data_root / 'test' / 'video_seqs' / 'demo'
        frames_dir = sequence_dir / 'frames'
        annos_dir = toolkit_root / 'annos'
        gt_dir = annos_dir / 'gt_rect'
        absent_dir = annos_dir / 'absent'
        frames_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        absent_dir.mkdir(parents=True)

        for frame_index in range(1, 3):
            _write_valid_png(frames_dir / f'{frame_index:05d}.png')
        (sequence_dir / 'language.txt').write_text('demo target\n',
                                                   encoding='utf-8')
        (toolkit_root / 'TNL2K_testing_set.txt').write_text('demo\n',
                                                            encoding='utf-8')
        (gt_dir / 'demo.txt').write_text('1,2,3,4\n0,0,0,0\n',
                                         encoding='utf-8')
        (absent_dir / 'demo.txt').write_text('0\n1\n', encoding='utf-8')

        (sequences, outside_count, unreadable_count, updated_gt,
         updated_text) = _testing_sequences(
            data_root, toolkit_root)

    assert list(sequences) == ['demo']
    np.testing.assert_array_equal(
        sequences['demo']['groundtruth'],
        np.asarray([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(
        sequences['demo']['absent'], np.asarray([False, True]))
    assert outside_count == 0
    assert unreadable_count == 0
    assert updated_gt == []
    assert updated_text == []


def test_read_absent_lenient_trims_extra_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'absent.txt'
        path.write_text('0\n1\n1\n', encoding='utf-8')
        absent = _read_absent_lenient(path, 2)

    np.testing.assert_array_equal(absent, np.asarray([False, True]))
