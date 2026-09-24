from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

if TYPE_CHECKING:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.variants.loss.instance.paper_base import (
        PaperBase,
    )


class LossMixer(nn.Module):
    """Combine multiple loss modules into a weighted sum."""

    def __init__(self, components: Sequence[Tuple[str, nn.Module, float]]):
        super().__init__()
        if not components:
            raise ValueError("LossMixer requires at least one component")

        names = [name for name, _, _ in components]
        if len(set(names)) != len(names):
            raise ValueError("LossMixer component names must be unique")

        self.components = nn.ModuleDict({name: module for name, module, _ in components})
        self.weights = {name: float(weight) for name, _, weight in components}

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        total = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        for name, module in self.components.items():
            weight = self.weights[name]
            if weight == 0:
                continue
            value = module(y_pred, y)
            total = total + weight * value
        return total


def integrate_deep_supervision(trainer: nnUNetTrainer, loss: nn.Module) -> nn.Module:
    """Wrap a loss with nnU-Net's deep supervision helper if required."""

    if not trainer.enable_deep_supervision:
        return loss

    deep_supervision_scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
    if trainer.is_ddp and not trainer._do_i_compile():
        weights[-1] = 1e-6
    else:
        weights[-1] = 0

    weights /= weights.sum()
    return DeepSupervisionWrapper(loss, weights)


def _build_global_dc_ce_loss(trainer: PaperBase) -> nn.Module:
    loss = DC_and_CE_loss(
        {
            "batch_dice": trainer.configuration_manager.batch_dice,
            "smooth": trainer.dice_smooth,
            "do_bg": False,
            "ddp": trainer.is_ddp,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=trainer.label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    if hasattr(loss, "dc") and trainer._do_i_compile():
        loss.dc = torch.compile(loss.dc)  # type: ignore[attr-defined]
    return loss
