from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from bicc_loss.blob import VectorizedBlobLoss
from bicc_loss.common import EmptyCellLoss
from bicc_loss.voronoi import (
    VectorizedBiCCLoss,
    VectorizedCCLoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.instance.loss_utils import (
    LossMixer,
    _build_global_dc_ce_loss,
    integrate_deep_supervision,
)
from nnunetv2.utilities.helpers import softmax_helper_dim1

InstanceLossKind = Literal["none", "blob", "cc", "bicc"]


class PaperBase(nnUNetTrainer):
    """Shared nnU-Net configuration for the BiCC paper experiments."""

    instance_loss_kind: InstanceLossKind
    alpha = 0.5
    # eps = 0 for CMB. Generally, it's best to leave it at 0, as higher values have no real upside
    # and might cause issues with low-fg datasets.
    dice_smooth = 0.0
    pred_branch_dice_smooth = 0.0
    # Use Eq. (3) to score cells without reference foreground.
    empty_cell_loss: EmptyCellLoss = "detached"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500

        if not hasattr(type(self), "instance_loss_kind"):
            raise TypeError(
                f"{type(self).__name__} must define class attribute "
                "'instance_loss_kind'"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    # CuPy's distance transforms support non-uniform spacing, so we supply the actual physical spacing.
    def _instance_loss_geometry(
        self,
    ) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
        sampling = tuple(float(i) for i in self.configuration_manager.spacing)
        patch_size = tuple(int(i) for i in self.configuration_manager.patch_size)
        self.print_to_log_file(
            "Instance Voronoi geometry: "
            f"sampling={sampling}, patch_size={patch_size}. "
            "Deep-supervision outputs scale sampling from their tensor shape."
        )
        return {"sampling": sampling, "patch_size": patch_size}

    def _build_global_loss(self) -> nn.Module:
        return _build_global_dc_ce_loss(self)

    def _build_instance_loss(self) -> nn.Module:
        if self.instance_loss_kind == "blob":
            return VectorizedBlobLoss(
                activation=softmax_helper_dim1,
                dice_smooth=self.dice_smooth,
            )
        geometry = self._instance_loss_geometry()
        if self.instance_loss_kind == "cc":
            return VectorizedCCLoss(
                dice_smooth=self.dice_smooth,
                **geometry,
            )
        if self.instance_loss_kind == "bicc":
            return VectorizedBiCCLoss(
                alpha=self.alpha,
                dice_smooth=self.dice_smooth,
                pred_branch_dice_smooth=self.pred_branch_dice_smooth,
                empty_cell_loss=self.empty_cell_loss,
                **geometry,
            )
        raise ValueError(
            f"{type(self).__name__} does not define an instance loss "
            f"(kind={self.instance_loss_kind!r})"
        )

    def _build_loss(self) -> nn.Module:  # type: ignore[override]
        if self.label_manager.has_regions:
            raise AssertionError(
                "Instance-level losses require label-based training without regions"
            )
        if self.label_manager.has_ignore_label:
            raise AssertionError("Instance-level losses do not support ignore labels")
        # We default to binary in this paper, but there's no reason you couldn't extend this
        # to > 2 classes (see Kundu et al., 2026).
        if self.label_manager.num_segmentation_heads != 2:
            raise AssertionError(
                "Blob/CC instance losses require a binary segmentation "
                "(background + foreground)"
            )

        global_loss = self._build_global_loss().to(self.device)
        if self.instance_loss_kind == "none":
            final_loss = global_loss
        else:
            instance_loss = self._build_instance_loss().to(self.device)
            final_loss = LossMixer(
                [("instance", instance_loss, 1.0), ("global", global_loss, 1.0)]
            )

        final_loss = integrate_deep_supervision(self, final_loss)
        final_loss = final_loss.to(self.device)

        # torch.compile is very important for performance here: each partition uses six
        # torch.scatter_add calls to compute per-component sums, which compilation can optimize.
        if self._do_i_compile():
            final_loss = torch.compile(final_loss)

        return final_loss
