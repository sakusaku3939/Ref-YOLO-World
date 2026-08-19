import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.speed.reliable_speed_measure import _expand_expression_items
from yolo_world.datasets.yolov5_refcoco import (
    CustomRefCOCODataset,
    CustomRefCOCOExpressionDataset)
from yolo_world.evaluation.metrics.custom_refcoco_metric import (
    CustomRefCOCOMetric)


class RefCOCOExpressionEvaluationTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        annotations = [
            {
                'image_id': 1,
                'ann_id': 10,
                'sentences': [
                    {'sent': 'left person'}, {'sent': 'person in red'}
                ],
                'raw_anns': json.dumps({'bbox': [10, 20, 30, 40]}),
                'bbox': [10, 20, 40, 60],
            },
            {
                'image_id': 2,
                'ann_id': 20,
                'sentences': [{'sent': 'right dog'}],
                'raw_anns': json.dumps({'bbox': [50, 60, 20, 10]}),
                'bbox': [50, 60, 70, 70],
            },
        ]
        self.ann_file = Path(self.temp_dir.name) / 'annotations.json'
        self.ann_file.write_text(json.dumps(annotations))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_expression_dataset_expands_every_caption(self):
        dataset = CustomRefCOCOExpressionDataset(
            ann_file=str(self.ann_file), pipeline=[], test_mode=True)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            [dataset.get_data_info(i)['texts'][0][0] for i in range(3)],
            ['left person', 'person in red', 'right dog'])
        self.assertEqual(
            [dataset.get_data_info(i)['img_id'] for i in range(3)],
            [0, 0, 1])
        self.assertEqual(
            [dataset.get_data_info(i)['expression_id'] for i in range(3)],
            [0, 1, 2])

    def test_expression_dataset_rejects_training_mode(self):
        with self.assertRaisesRegex(ValueError, 'evaluation-only'):
            CustomRefCOCOExpressionDataset(
                ann_file=str(self.ann_file), pipeline=[], test_mode=False)

    def test_reference_dataset_keeps_one_sample_per_reference(self):
        dataset = CustomRefCOCODataset(
            ann_file=str(self.ann_file), pipeline=[], test_mode=True)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            [dataset.get_data_info(i)['texts'][0][0] for i in range(2)],
            ['left person', 'right dog'])
        self.assertEqual(
            [dataset.get_data_info(i)['expression_id'] for i in range(2)],
            [0, 1])

    def test_metric_validates_expression_count(self):
        metric = CustomRefCOCOMetric(
            ann_file=str(self.ann_file), expected_num_expressions=3)

        computed = metric.compute_metrics(self._expression_results())

        self.assertEqual(computed['Acc@0.5'], 100.0)
        self.assertEqual(computed['n_total'], 3)

    def test_metric_rejects_legacy_ref_count(self):
        metric = CustomRefCOCOMetric(
            ann_file=str(self.ann_file), expected_num_expressions=3)
        legacy_results = self._expression_results()[1:]

        with self.assertRaisesRegex(ValueError, 'expected 3 expressions'):
            metric.compute_metrics(legacy_results)

    def test_metric_deduplicates_distributed_expression_results(self):
        metric = CustomRefCOCOMetric(
            ann_file=str(self.ann_file), expected_num_expressions=3)
        results = self._expression_results()
        duplicated = [
            {**result, 'expression_id': expression_id}
            for expression_id, result in enumerate(results)
            for _ in range(2)
        ]

        computed = metric.compute_metrics(duplicated)

        self.assertEqual(computed['Acc@0.5'], 100.0)
        self.assertEqual(computed['n_total'], 3)

    def test_speed_measure_expands_every_expression(self):
        grouped_items = [
            {
                'image_path': 'one.jpg',
                'captions': ['left person', 'person in red'],
            },
            {
                'image_path': 'two.jpg',
                'captions': ['right dog'],
            },
        ]

        expressions = _expand_expression_items(grouped_items)

        self.assertEqual(len(expressions), 3)
        self.assertEqual(
            [item['captions'][0] for item in expressions],
            ['left person', 'person in red', 'right dog'])
        self.assertEqual(
            [item['_ref_index'] for item in expressions], [0, 0, 1])
        self.assertEqual(
            [item['_expression_id'] for item in expressions], [0, 1, 2])

    @staticmethod
    def _expression_results():
        return [
            {
                'img_id': 0,
                'bboxes': np.array([[10, 20, 40, 60]]),
                'scores': np.array([1.0]),
                'labels': np.array([0]),
            },
            {
                'img_id': 0,
                'bboxes': np.array([[10, 20, 40, 60]]),
                'scores': np.array([1.0]),
                'labels': np.array([0]),
            },
            {
                'img_id': 1,
                'bboxes': np.array([[50, 60, 70, 70]]),
                'scores': np.array([1.0]),
                'labels': np.array([0]),
            },
        ]


if __name__ == '__main__':
    unittest.main()
