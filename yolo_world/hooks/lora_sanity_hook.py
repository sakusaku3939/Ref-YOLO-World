"""Sanity checks for CLIP text LoRA experiments."""

from __future__ import annotations

import torch

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LoRASanityHook(Hook):
    """Validate that text LoRA adapters are injected and initially no-op."""

    def __init__(
        self,
        expected_pairs: int = 8,
        zero_tolerance: float = 1e-8,
        priority: str = "VERY_HIGH",
    ) -> None:
        self.expected_pairs = expected_pairs
        self.zero_tolerance = zero_tolerance
        self.priority = priority
        self._checked_grad = False

    def before_train(self, runner) -> None:
        pairs = self._collect_lora_pairs(runner.model)
        if len(pairs) != self.expected_pairs:
            raise RuntimeError(
                f"[LoRASanityHook] expected {self.expected_pairs} LoRA pairs, "
                f"found {len(pairs)}"
            )

        max_b_abs = 0.0
        for name, lora_a, lora_b in pairs:
            if not lora_a.requires_grad or not lora_b.requires_grad:
                raise RuntimeError(
                    f"[LoRASanityHook] LoRA params are frozen for {name}"
                )
            max_b_abs = max(max_b_abs, float(lora_b.detach().abs().max().cpu()))

        if max_b_abs > self.zero_tolerance:
            raise RuntimeError(
                f"[LoRASanityHook] LoRA B is not zero-init: max={max_b_abs:.6g}"
            )

        runner.logger.info(
            f"[LoRASanityHook] OK: pairs={len(pairs)}, "
            f"max_lora_B_abs={max_b_abs:.6g}"
        )

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        if self._checked_grad:
            return
        self._checked_grad = True

        grad_norm = 0.0
        finite = True
        seen_grad = False
        for _, lora_a, lora_b in self._collect_lora_pairs(runner.model):
            for param in (lora_a, lora_b):
                if param.grad is None:
                    continue
                seen_grad = True
                grad = param.grad.detach()
                finite = finite and bool(torch.isfinite(grad).all().item())
                grad_norm += float(grad.norm().cpu())

        if seen_grad:
            runner.logger.info(
                f"[LoRASanityHook] first observed LoRA grad_norm={grad_norm:.6g}, "
                f"finite={finite}"
            )
        else:
            runner.logger.warning(
                "[LoRASanityHook] LoRA gradients were not visible at "
                "after_train_iter; optimizer timing may have cleared them."
            )

    @staticmethod
    def _collect_lora_pairs(model):
        pairs = []
        module = model.module if hasattr(model, "module") else model
        for name, submodule in module.named_modules():
            if hasattr(submodule, "lora_A") and hasattr(submodule, "lora_B"):
                pairs.append((name, submodule.lora_A, submodule.lora_B))
        return pairs
