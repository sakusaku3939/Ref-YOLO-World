"""Hook for measuring inference time (total, per image, and aggregated over all
processes in distributed runs)."""
import json
import os
import os.path as osp
import time

import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class InferenceTimeHook(Hook):
    """Measure the time of the whole test inference step (data loading, forward, results).

    Computes the per-image time from the total inference time over all iterations and
    the total number of images.
    In distributed runs, times and image counts are aggregated over all processes with
    all_reduce(sum).
    Results are written to stdout and to inference_time.json under work_dir.
    """

    def before_test_epoch(self, runner):
        self._total_time = 0.0
        self._total_images = 0
        self._iter_start = None
        self._saved_inference_time_path = None

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        self._iter_start = time.perf_counter()

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self._iter_start is None:
            return
        elapsed = time.perf_counter() - self._iter_start
        self._total_time += elapsed
        batch_size = 0
        if data_batch is not None and isinstance(data_batch, dict):
            samples = data_batch.get('data_samples')
            if samples is not None:
                batch_size = len(samples)
        self._total_images += batch_size
        self._iter_start = None

    def after_test_epoch(self, runner, metrics=None):
        total_time = self._total_time
        total_images = self._total_images

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = next(runner.model.parameters()).device
            time_t = torch.tensor([total_time], dtype=torch.float64, device=device)
            count_t = torch.tensor([float(total_images)], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(time_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count_t, op=torch.distributed.ReduceOp.SUM)
            total_time = time_t.item()
            total_images = int(count_t.item())

        time_per_image_ms = (total_time / total_images * 1000.0) if total_images > 0 else 0.0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

        result = {
            'total_inference_time_sec': round(total_time, 4),
            'total_images': total_images,
            'time_per_image_ms': round(time_per_image_ms, 4),
            'world_size': world_size,
        }

        rank = (
            torch.distributed.get_rank()
            if (torch.distributed.is_available() and torch.distributed.is_initialized())
            else 0
        )
        if rank == 0:
            msg = (
                f"[InferenceTime] total_inference_time_sec={result['total_inference_time_sec']}, "
                f"total_images={result['total_images']}, "
                f"time_per_image_ms={result['time_per_image_ms']}, "
                f"world_size={result['world_size']}"
            )
            if hasattr(runner, 'logger') and runner.logger is not None:
                runner.logger.info(msg)

            out_dir = getattr(runner, 'work_dir', None)
            if out_dir:
                out_path = osp.join(out_dir, 'inference_time.json')
                os.makedirs(out_dir, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                self._saved_inference_time_path = out_path

    def after_run(self, runner):
        """Print in after_run so that 'Inference time saved to' appears after the Epoch(test) log."""
        rank = (
            torch.distributed.get_rank()
            if (torch.distributed.is_available() and torch.distributed.is_initialized())
            else 0
        )
        if rank == 0 and getattr(self, '_saved_inference_time_path', None):
            if hasattr(runner, 'logger') and runner.logger is not None:
                runner.logger.info(f"Inference time saved to {self._saved_inference_time_path}")
            self._saved_inference_time_path = None
