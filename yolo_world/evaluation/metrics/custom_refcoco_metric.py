# Copyright (c) Tencent Inc. All rights reserved.
import json
import os
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger

from mmyolo.registry import METRICS


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


@METRICS.register_module()
class CustomRefCOCOMetric(BaseMetric):
    """RefCOCO Metric for Referring Expression Comprehension.

    Computes Accuracy@IoU threshold (default 0.5).
    For each sample, if the top-1 predicted box has IoU >= threshold
    with the ground truth box, it's considered correct.
    
    Note: HuggingFace版RefCOCOデータセット（jxu124/refcoco*）では、
    トップレベルの'bbox'フィールドは既に[x1, y1, x2, y2]形式に変換されています。
    一方、'raw_anns'内の'bbox'はCOCO標準の[x, y, w, h]形式のままです。
    """

    default_prefix: Optional[str] = 'refcoco'

    def __init__(
            self,
            ann_file: str,
            iou_threshold: float = 0.5,
            format_only: bool = False,
            outfile_prefix: Optional[str] = None,
            backend_args: Optional[dict] = None,
            collect_device: str = 'cpu',
            prefix: Optional[str] = None,
            dump_results: bool = False,
            results_dir: Optional[str] = None,
            expected_num_expressions: Optional[int] = None,
            dump_topk: int = 0,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.iou_threshold = iou_threshold
        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.backend_args = backend_args
        self.dump_results = dump_results
        self.expected_num_expressions = expected_num_expressions
        self.dump_topk = max(0, int(dump_topk))
        
        if dump_results:
            self.results_dir = osp.abspath(results_dir or './refcoco_results')
            os.makedirs(self.results_dir, exist_ok=True)

        # Load ground truth annotations
        with get_local_path(ann_file, backend_args=self.backend_args) as p:
            with open(p, 'r') as f:
                self.gt_data = json.load(f)
        
        # Build GT map: img_id -> gt_info
        self.gt_map = {}
        for idx, item in enumerate(self.gt_data):
            self.gt_map[idx] = item

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for ds in data_samples:
            pred = ds['pred_instances']
            
            result = dict(
                img_id=ds['img_id'],
                expression_id=ds.get('expression_id'),
                bboxes=pred['bboxes'].cpu().numpy(),
                scores=pred['scores'].cpu().numpy(),
                labels=pred['labels'].cpu().numpy(),
            )
            
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute RefCOCO metrics.

        Args:
            results: List of prediction results.

        Returns:
            Dict containing evaluation metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        eval_results = OrderedDict()
        
        if self.format_only:
            if self.outfile_prefix:
                dump(results, f'{self.outfile_prefix}_predictions.json')
                logger.info(f'Saved predictions to {self.outfile_prefix}_predictions.json')
            return eval_results

        # Distributed evaluation may pad or repeat samples. Keep exactly one
        # prediction per expression before validating the formal sample count.
        unique_results = []
        seen_expression_ids = set()
        for pred in results:
            expression_id = pred.get('expression_id')
            if expression_id is None:
                unique_results.append(pred)
                continue
            expression_id = int(expression_id)
            if expression_id in seen_expression_ids:
                continue
            seen_expression_ids.add(expression_id)
            unique_results.append(pred)

        # Compute Accuracy@IoU
        n_correct = 0
        n_total = 0
        
        detailed_results = []
        
        for pred in unique_results:
            img_id = pred['img_id']
            
            if img_id not in self.gt_map:
                continue
            
            gt_item = self.gt_map[img_id]
            gt_bbox = gt_item.get('bbox', [])

            if len(pred['bboxes']) == 0:
                top1_box = None
                top1_iou = 0.0
                top1_idx = None
            else:
                top1_idx = np.argmax(pred['scores'])
                top1_box = pred['bboxes'][top1_idx]
                top1_iou = 0.0

            is_correct = False
            if len(gt_bbox) == 4:
                # HuggingFace版RefCOCOはxyxy、COCO由来データはxywh。
                dataset_source = gt_item.get('dataset_source', '')
                if dataset_source == 'coco':
                    x, y, w, h = gt_bbox
                    gt_box = np.array([x, y, x + w, y + h])
                else:
                    gt_box = np.array(gt_bbox)

                n_total += 1
                if top1_box is not None:
                    top1_iou = compute_iou(top1_box, gt_box)
                    is_correct = top1_iou >= self.iou_threshold
                if is_correct:
                    n_correct += 1
            
            if self.dump_results:
                topk_indices = np.argsort(pred['scores'])[::-1][:self.dump_topk] if self.dump_topk > 0 else []
                serialized_gt_bbox = (
                    [float(x) for x in gt_bbox]
                    if len(gt_bbox) == 4
                    and all(np.isscalar(x) for x in gt_bbox)
                    else [])
                detailed_results.append(dict(
                    img_id=int(img_id),
                    expression_id=(
                        int(pred['expression_id'])
                        if pred.get('expression_id') is not None else None),
                    image_path=str(gt_item.get('image_path', '')),
                    captions=list(gt_item.get('captions', [])),
                    gt_bbox=serialized_gt_bbox,
                    pred_bbox=top1_box.tolist() if top1_box is not None else [],
                    pred_score=(
                        float(pred['scores'][top1_idx])
                        if top1_idx is not None else 0.0),
                    iou=float(top1_iou),
                    is_correct=bool(is_correct),
                    all_pred_bboxes=[pred['bboxes'][i].tolist() for i in topk_indices],
                    all_pred_scores=[float(pred['scores'][i]) for i in topk_indices],
                    all_pred_labels=[int(pred['labels'][i]) for i in topk_indices],
                ))
        
        # Calculate accuracy
        if (self.expected_num_expressions is not None
                and n_total != self.expected_num_expressions):
            raise ValueError(
                'RefCOCO evaluation sample count mismatch: '
                f'expected {self.expected_num_expressions} expressions, '
                f'but evaluated {n_total}.')

        accuracy = n_correct / n_total if n_total > 0 else 0.0
        
        eval_results['Acc@0.5'] = round(accuracy * 100, 2)
        eval_results['n_correct'] = n_correct
        eval_results['n_total'] = n_total
        
        logger.info(f'RefCOCO Evaluation Results:')
        logger.info(f'  Accuracy@{self.iou_threshold}: {accuracy * 100:.2f}%')
        logger.info(f'  Correct: {n_correct} / {n_total}')
        
        # Dump detailed results if requested
        if self.dump_results and detailed_results:
            out_path = osp.join(self.results_dir, 'detailed_results.json')
            with open(out_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f'Detailed results saved to {out_path}')
        
        return eval_results
