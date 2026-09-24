import torch

from bicc_loss.common import VectorizedLossMode, softmax_helper_dim1
from bicc_loss.reductions import (
    _prepare_vectorized_loss_inputs,
    _vectorized_blob_reduction,
)
from bicc_loss.voronoi import _BaseVectorizedLoss
from bicc_loss.components import get_cc


class VectorizedBlobLoss(_BaseVectorizedLoss):
    """Vectorized blob loss for binary instance segmentation."""

    def __init__(
        self,
        activation=softmax_helper_dim1,
        dice_smooth: float = 1e-5,
        mode: VectorizedLossMode = "dice_ce",
    ) -> None:
        super().__init__(activation=activation, mode=mode, dice_smooth=dice_smooth)

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        connected_components = get_cc(y, do_bg=False)
        return connected_components[:, 0, ...]

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
        return _vectorized_blob_reduction(
            foreground_pred,
            foreground_true,
            ce_map,
            components,
            self.mode,
            num_component_slots,
            self.dice_smooth,
        )
