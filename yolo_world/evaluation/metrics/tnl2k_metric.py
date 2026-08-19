"""Frame-level validation metric for TNL2K fine-tuning."""

import json
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from mmyolo.registry import METRICS


def _iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@METRICS.register_module()
class TNL2KFrameMetric(BaseMetric):
    """Compute top-1 frame Acc@0.5 from a TNL2K JSONL manifest."""

    default_prefix: Optional[str] = 'tnl2k'

    def __init__(self,
                 ann_file: str,
                 iou_threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_threshold = iou_threshold
        self.gt_map = {}
        with get_local_path(ann_file) as local_path:
            with open(local_path, encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    x, y, width, height = item['bbox_xywh']
                    self.gt_map[int(item['img_id'])] = np.asarray(
                        [x, y, x + width, y + height], dtype=np.float64)
        if not self.gt_map:
            raise ValueError(f'No annotations in {ann_file}')

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            prediction = sample['pred_instances']
            self.results.append({
                'img_id': int(sample['img_id']),
                'bboxes': prediction['bboxes'].cpu().numpy(),
                'scores': prediction['scores'].cpu().numpy(),
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        correct = 0
        seen = set()
        for result in results:
            image_id = result['img_id']
            if image_id not in self.gt_map:
                raise ValueError(f'Unexpected image id: {image_id}')
            if image_id in seen:
                raise ValueError(f'Duplicate image id: {image_id}')
            seen.add(image_id)
            if len(result['scores']):
                index = int(np.argmax(result['scores']))
                iou = _iou_xyxy(
                    result['bboxes'][index], self.gt_map[image_id])
                correct += int(iou >= self.iou_threshold)
        if len(seen) != len(self.gt_map):
            raise ValueError(
                f'Evaluated {len(seen)} of {len(self.gt_map)} manifest rows')
        accuracy = correct / len(self.gt_map)
        return OrderedDict({
            'Acc@0.5': round(accuracy * 100.0, 2),
            'n_correct': correct,
            'n_total': len(self.gt_map),
        })
