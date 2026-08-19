# Copyright (c) Tencent Inc. All rights reserved.
import json
import os.path as osp
import random
from copy import deepcopy
from typing import List, Union

from mmengine.fileio import get_local_path
from mmengine.dataset import BaseDataset
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class CustomRefCOCODataset(BaseDataset):
    """RefCOCO dataset for referring expression comprehension.

    Each sample exposes one selected referring expression with label 0. This
    dataset does not add semantic negative expressions; fixed-width empty
    slots, when requested, are added later by ``RandomLoadText``.
    
    Supports the original RefCOCO JSON format with fields:
    - file_name: image filename (e.g., "COCO_train2014_000000581857_16.jpg")
    - image_id: COCO image ID
    - sentences: list of referring expressions
    - raw_anns: JSON string containing bbox and segmentation
    
    The dataset expects COCO train2014 images to be available.
    """

    METAINFO = {
        'classes': ('object',),
        'palette': [(220, 20, 60)]
    }

    def __init__(self,
                 ann_file: str = '',
                 img_prefix: str = '',
                 data_root: str = '',
                 pipeline: List = [],
                 test_mode: bool = False,
                 filter_cfg: dict = None,
                 backend_args: dict = None,
                 **kwargs):
        self.img_prefix = img_prefix
        self.filter_cfg = filter_cfg
        self.backend_args = backend_args
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs
        )

    def load_data_list(self) -> List[dict]:
        """Load annotations from RefCOCO JSON file.

        Returns:
            List[dict]: A list of annotation.
        """
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                raw_data = json.load(f)

        data_list = []
        for idx, item in enumerate(raw_data):
            data_info = self.parse_data_info(item, idx)
            if data_info is not None:
                data_list.append(data_info)

        if self.filter_cfg is not None:
            data_list = self._filter_data(data_list)

        return data_list

    def parse_data_info(self, raw_item: dict, idx: int) -> Union[dict, None]:
        """Parse raw annotation to target format.

        Handles original RefCOCO format with:
        - file_name: image filename
        - image_id: COCO image ID  
        - sentences: list of dicts with 'sent' key
        - raw_anns: JSON string with bbox in [x, y, w, h] format

        Args:
            raw_item (dict): Raw data information from RefCOCO JSON
            idx (int): Index of the item

        Returns:
            Union[dict, None]: Parsed annotation or None if invalid.
        """
        data_info = {}

        image_id = raw_item.get('image_id', 0)
        file_name = f'COCO_train2014_{image_id:012d}.jpg'
        
        if self.img_prefix:
            img_path = osp.join(self.img_prefix, file_name)
        elif self.data_root:
            img_path = osp.join(self.data_root, 'coco', 'train2014', file_name)
        else:
            img_path = osp.join('coco', 'train2014', file_name)
        
        data_info['img_path'] = img_path
        data_info['img_id'] = idx
        data_info['ann_id'] = raw_item.get('ann_id', idx)
        if self.test_mode:
            # Make one-reference-per-sample evaluation deterministic and
            # deduplicatable when a distributed sampler pads the dataset.
            data_info['expression_id'] = idx

        raw_anns = raw_item.get('raw_anns', '{}')
        if isinstance(raw_anns, str):
            try:
                anns = json.loads(raw_anns)
            except json.JSONDecodeError:
                return None
        else:
            anns = raw_anns
        
        bbox_xywh = anns.get('bbox', [])
        if len(bbox_xywh) != 4:
            return None
        
        x, y, w, h = bbox_xywh
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        if x2 <= x1 or y2 <= y1:
            return None

        data_info['height'] = 480
        data_info['width'] = 640

        sentences = raw_item.get('sentences', [])
        if not sentences:
            return None
        
        captions = []
        for sent_item in sentences:
            if isinstance(sent_item, dict):
                sent_text = sent_item.get('sent', sent_item.get('raw', ''))
            else:
                sent_text = str(sent_item)
            if sent_text:
                captions.append(sent_text)
        
        if not captions:
            return None

        data_info['all_captions'] = captions
        selected_caption = captions[0]
        data_info['texts'] = [[selected_caption]]

        instance = {
            'bbox': [x1, y1, x2, y2],
            'bbox_label': 0,
            'ignore_flag': 0
        }
        data_info['instances'] = [instance]
        data_info['is_detection'] = 1

        return data_info

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.
        
        For training, select one caption from the equivalent expressions.
        The alternatives describe the same target and are not negative texts.
        """
        data_info = super().get_data_info(idx)
        
        if not self.test_mode and 'all_captions' in data_info:
            captions = data_info['all_captions']
            selected_caption = random.choice(captions)
            data_info['texts'] = [[selected_caption]]
        
        return data_info

    def _filter_data(self, data_list: List[dict]) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for data_info in data_list:
            width = int(data_info['width'])
            height = int(data_info['height'])
            if filter_empty_gt and len(data_info.get('instances', [])) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos


@DATASETS.register_module()
class CustomRefCOCOExpressionDataset(CustomRefCOCODataset):
    """RefCOCO evaluation dataset with one sample per expression.

    RefCOCO annotations group multiple equivalent referring expressions under
    one target object. Standard evaluation treats every expression as an
    independent sample, while :class:`CustomRefCOCODataset` intentionally
    keeps the historical one-sample-per-target behavior for reproducibility.

    ``img_id`` remains the source annotation index so that
    ``CustomRefCOCOMetric`` can reuse the target box for every expression.
    Dataset indices remain unique and are used by the sampler.
    """

    def __init__(self, *args, test_mode: bool = True, **kwargs):
        if not test_mode:
            raise ValueError(
                'CustomRefCOCOExpressionDataset is evaluation-only; '
                'set test_mode=True.')
        super().__init__(*args, test_mode=True, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load and expand annotations to one item per expression."""
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                raw_data = json.load(f)

        data_list = []
        for ref_idx, item in enumerate(raw_data):
            ref_info = self.parse_data_info(item, ref_idx)
            if ref_info is None:
                continue

            captions = ref_info.pop('all_captions')
            for expression_index, caption in enumerate(captions):
                data_info = deepcopy(ref_info)
                data_info['texts'] = [[caption]]
                data_info['expression_id'] = len(data_list)
                data_info['expression_index'] = expression_index
                data_list.append(data_info)

        if self.filter_cfg is not None:
            data_list = self._filter_data(data_list)

        return data_list
