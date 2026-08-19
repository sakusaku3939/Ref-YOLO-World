"""Frame-level dataset adapter for TNL2K fine-tuning."""

import json
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class TNL2KFrameDataset(BaseDataset):
    """Load a JSONL manifest with one referring-expression sample per frame."""

    METAINFO = {
        'classes': ('object',),
        'palette': [(220, 20, 60)],
    }

    def __init__(self, backend_args: dict = None, **kwargs):
        self.backend_args = backend_args
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    bbox = item['bbox_xywh']
                    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        raise ValueError(
                            f'Invalid bbox at {local_path}:{line_number}')
                    image_path = item['img_path']
                    if self.data_root and not osp.isabs(image_path):
                        image_path = osp.join(self.data_root, image_path)
                    x, y, width, height = bbox
                    data_list.append({
                        'img_id': int(item['img_id']),
                        'img_path': image_path,
                        'sequence': item['sequence'],
                        'frame_index': int(item['frame_index']),
                        'texts': [[item['expression']]],
                        'is_detection': 1,
                        'instances': [{
                            'bbox': [x, y, x + width, y + height],
                            'bbox_label': 0,
                            'ignore_flag': 0,
                        }],
                    })
        if not data_list:
            raise ValueError(f'No samples in TNL2K manifest: {self.ann_file}')
        return data_list
