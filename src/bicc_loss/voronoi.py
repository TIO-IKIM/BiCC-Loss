from typing import Optional, Sequence

import torch

from bicc_loss.common import (
    EmptyCellLoss,
    VectorizedLossMode,
    _normalize_patch_size,
    _normalize_sampling,
    softmax_helper_dim1,
)
from bicc_loss.reductions import (
    _prepare_vectorized_loss_inputs,
    _vectorized_cc_reduction,
)
from bicc_loss.components import get_voronoi


class _BaseVectorizedLoss(torch.nn.Module):
    """Specialized reduction path that avoids per-component masking loops."""

    def __init__(
        self,
        activation,
        mode: VectorizedLossMode,
        dice_smooth: float = 1e-5,
        sampling: Optional[Sequence[float]] = None,
        patch_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.activation = activation
        if mode not in ("dice", "ce", "dice_ce"):
            raise ValueError(f"Unsupported vectorized loss mode: {mode}")
        self.mode = mode
        if dice_smooth < 0.0:
            raise ValueError(f"dice_smooth must be non-negative, got {dice_smooth}")
        self.dice_smooth = float(dice_smooth)
        self.sampling = _normalize_sampling(sampling)
        self.patch_size = _normalize_patch_size(patch_size)
        if (
            self.sampling is not None
            and self.patch_size is not None
            and len(self.sampling) != len(self.patch_size)
        ):
            raise ValueError("sampling and patch_size must have the same number of entries")

    def forward(self, y_pred, y):
        self._validate_inputs(y_pred, y)

        target_fg = y == 1
        components = self._compute_components(target_fg)

        self._validate_components(y_pred, y, components)
        return self._vectorized_loss(y_pred, y, components)

    def _validate_inputs(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        assert (
            y_pred.ndim == 5 and y_pred.shape[1] == 2
        ), f"Expected y_pred with shape [B,2,H,W,D], but got {tuple(y_pred.shape)}"
        assert list(y.shape) == [
            y_pred.shape[0],
            1,
            *y_pred.shape[2:],
        ], f"Expected y with shape [B,1,H,W,D], but got {tuple(y.shape)}"
        assert y.dtype == torch.int16, f"Expected y.dtype=torch.int16, but got {y.dtype}"
        assert y.device == y_pred.device, (
            f"y and y_pred must reside on the same device, "
            f"but got y on {y.device} and y_pred on {y_pred.device}"
        )

    def _validate_components(
        self, y_pred: torch.Tensor, y: torch.Tensor, components: torch.Tensor
    ) -> None:
        assert components.dtype == torch.int64
        assert y.shape[0] == y_pred.shape[0], "Batch size mismatch between y_pred and y"
        assert y.shape[2:] == y_pred.shape[2:], "Spatial shape mismatch between y_pred and y"
        assert y.dtype == torch.int16
        expected_shape = [y_pred.shape[0], *y_pred.shape[2:]]
        assert (
            list(components.shape) == expected_shape
        ), f"Expected connected components with shape [B,H,W,D], but got {tuple(components.shape)}"

    def _vectorized_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        components: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # Deep supervision has multiple sizes so we need to adjust our spacing for that.
    def _effective_sampling(
        self, spatial_shape: Sequence[int]
    ) -> Optional[tuple[float, ...]]:
        spatial_shape = tuple(int(i) for i in spatial_shape)
        if self.sampling is None:
            return None
        if len(self.sampling) != len(spatial_shape):
            raise ValueError(
                f"Expected {len(spatial_shape)} sampling values, got {len(self.sampling)}"
            )
        if self.patch_size is None:
            return self.sampling
        if len(self.patch_size) != len(spatial_shape):
            raise ValueError(
                f"Expected {len(spatial_shape)} patch size values, got {len(self.patch_size)}"
            )
        return tuple(
            spacing * (full_size / current_size)
            for spacing, full_size, current_size in zip(
                self.sampling, self.patch_size, spatial_shape
            )
        )


class VectorizedCCLoss(_BaseVectorizedLoss):
    """Vectorized Voronoi connected-component loss for binary instances."""

    def __init__(
        self,
        dice_smooth: float = 0.0,
        mode: VectorizedLossMode = "dice_ce",
        sampling: Optional[Sequence[float]] = None,
        patch_size: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            activation=softmax_helper_dim1,
            mode=mode,
            dice_smooth=dice_smooth,
            sampling=sampling,
            patch_size=patch_size,
        )

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        voronoi = get_voronoi(
            y,
            do_bg=True,
            sampling=self._effective_sampling(y.shape[2:]),
        )
        return voronoi[:, 0, ...]

    def _vectorized_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        components: torch.Tensor,
    ) -> torch.Tensor:
        foreground_pred, foreground_true, ce_map = _prepare_vectorized_loss_inputs(
            y_pred, y, self.activation
        )
        num_component_slots = max(1, int(components.max().item()))
        return _vectorized_cc_reduction(
            foreground_pred,
            foreground_true,
            ce_map,
            components,
            self.mode,
            num_component_slots,
            dice_smooth=self.dice_smooth,
        )


class VectorizedBiCCLoss(_BaseVectorizedLoss):
    """Bidirectional Voronoi connected-component loss."""

    def __init__(
        self,
        dice_smooth: float = 0.0,
        mode: VectorizedLossMode = "dice_ce",
        segmentation_threshold: float = 0.5,
        alpha: float = 0.5,
        sampling: Optional[Sequence[float]] = None,
        patch_size: Optional[Sequence[int]] = None,
        pred_branch_dice_smooth: Optional[float] = None,
        empty_cell_loss: EmptyCellLoss = "detached",
    ) -> None:
        super().__init__(
            activation=softmax_helper_dim1,
            mode=mode,
            dice_smooth=dice_smooth,
            sampling=sampling,
            patch_size=patch_size,
        )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.segmentation_threshold = segmentation_threshold
        self.alpha = float(alpha)
        if pred_branch_dice_smooth is not None and pred_branch_dice_smooth < 0.0:
            raise ValueError(
                "pred_branch_dice_smooth must be non-negative, got "
                f"{pred_branch_dice_smooth}"
            )
        self.pred_branch_dice_smooth = (
            self.dice_smooth
            if pred_branch_dice_smooth is None
            else float(pred_branch_dice_smooth)
        )
        # Choose how to score cells without reference foreground.
        if empty_cell_loss not in ("dice", "detached"):
            raise ValueError(f"Unsupported empty_cell_loss: {empty_cell_loss!r}")
        self.empty_cell_loss: EmptyCellLoss = empty_cell_loss

    def forward(self, y_pred, y):
        self._validate_inputs(y_pred, y)

        target_fg = y == 1
        spatial_shape = y_pred.shape[2:]
        probs = self.activation(y_pred) if self.activation is not None else y_pred
        pred_fg = self._prediction_foreground_mask(probs[:, 1:2])
        joint_foreground = torch.cat((target_fg, pred_fg), dim=0)
        joint_voronoi = self._voronoi_components(
            joint_foreground,
            spatial_shape,
        )
        gt_voronoi, pred_voronoi = joint_voronoi.chunk(2, dim=0)

        self._validate_components(y_pred, y, gt_voronoi)
        self._validate_components(y_pred, y, pred_voronoi)

        # Both directions share probabilities and CE but use different partitions.
        num_slots = max(1, int(joint_voronoi.max().item()))
        foreground_pred, foreground_true, ce_map = _prepare_vectorized_loss_inputs(
            y_pred, y, self.activation
        )

        def cc_loss(components: torch.Tensor, dice_smooth: float) -> torch.Tensor:
            return _vectorized_cc_reduction(
                foreground_pred,
                foreground_true,
                ce_map,
                components,
                self.mode,
                num_slots,
                dice_smooth,
                self.empty_cell_loss,
            )

        return (1.0 - self.alpha) * cc_loss(
            gt_voronoi, self.dice_smooth
        ) + self.alpha * cc_loss(pred_voronoi, self.pred_branch_dice_smooth)

    def _voronoi_components(
        self,
        foreground_mask: torch.Tensor,
        spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        sampling = self._effective_sampling(spatial_shape)
        voronoi = get_voronoi(
            foreground_mask,
            do_bg=True,
            sampling=sampling,
        )
        return voronoi[:, 0, ...]

    def _prediction_foreground_mask(self, foreground_prob: torch.Tensor) -> torch.Tensor:
        return foreground_prob > self.segmentation_threshold
