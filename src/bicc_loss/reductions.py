from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.func import vmap

from bicc_loss.common import (
    EmptyCellLoss,
    VectorizedLossMode,
    configure_instance_loss_dynamo,
)


configure_instance_loss_dynamo()

_DIVISION_FLOOR = 1e-8

# Spread additions over 32 replicas to reduce contention within each GPU warp.
_NUM_REPLICAS = 32


def _replica_count(num_voxels: int) -> int:
    """Find the largest replica count that divides the volume."""
    replicas = _NUM_REPLICAS
    while num_voxels % replicas:
        replicas //= 2
    return replicas


def _combine_vectorized_loss_terms(
    *,
    mode: VectorizedLossMode,
    intersection: torch.Tensor,
    pred_sum: torch.Tensor,
    true_sum: torch.Tensor,
    ce_sum: torch.Tensor,
    count: torch.Tensor,
    weighted_pred_sum: Optional[torch.Tensor] = None,
    dice_smooth: float = 0.0,
    empty_cell_loss: EmptyCellLoss = "detached",
) -> torch.Tensor:
    count_min = 1.0
    normal_dice_loss = 1.0 - (
        (2.0 * intersection + dice_smooth)
        / (pred_sum + true_sum + dice_smooth).clamp_min(_DIVISION_FLOOR)
    )
    if empty_cell_loss == "dice":
        dice_loss = normal_dice_loss
    else:
        if weighted_pred_sum is None:
            raise ValueError("Detached empty-cell loss requires weighted predictions")
        empty_target_loss = weighted_pred_sum / pred_sum.detach().clamp_min(
            _DIVISION_FLOOR
        )
        dice_loss = torch.where(true_sum > 0, normal_dice_loss, empty_target_loss)
    if mode == "dice":
        return dice_loss
    if mode == "ce":
        return ce_sum / count.clamp_min(count_min)

    ce_loss = ce_sum / count.clamp_min(count_min)
    return dice_loss + ce_loss


def _whole_volume_fallback_loss(
    *,
    mode: VectorizedLossMode,
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: Optional[torch.Tensor] = None,
    smooth: float,
) -> torch.Tensor:
    if mode == "ce":
        assert ce_map is not None
        return ce_map.mean()

    intersection = (foreground_pred * foreground_true).sum()
    denominator = foreground_pred.sum() + foreground_true.sum()
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth).clamp_min(
        _DIVISION_FLOOR
    )
    if mode == "dice":
        return dice_loss

    assert ce_map is not None
    return dice_loss + ce_map.mean()


def _prepare_vectorized_loss_inputs(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    activation: Optional[Callable],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = activation(y_pred) if activation is not None else y_pred
    foreground_pred = probs[:, 1].to(torch.float32)
    foreground_true = (y[:, 0] == 1).to(torch.float32)
    # Cross entropy requires int64 class indices.
    ce_map = F.cross_entropy(y_pred.to(torch.float32), y[:, 0].long(), reduction="none")
    return foreground_pred, foreground_true, ce_map


def _vectorized_cc_reduction(
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: torch.Tensor,
    components: torch.Tensor,
    mode: VectorizedLossMode,
    num_component_slots: int,
    dice_smooth: float = 1e-5,
    empty_cell_loss: EmptyCellLoss = "detached",
) -> torch.Tensor:
    num_voxels = foreground_pred[0].numel()
    replicas = _replica_count(num_voxels)

    def per_sample_loss(
        sample_foreground_pred: torch.Tensor,
        sample_foreground_true: torch.Tensor,
        sample_ce_map: torch.Tensor,
        sample_components: torch.Tensor,
    ) -> torch.Tensor:
        num_slots_with_bg = num_component_slots + 1
        # Each column accumulates into a separate replica.
        replica_ids = sample_components.reshape(-1, replicas)

        def segment_sum(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(
                num_slots_with_bg, replicas, device=values.device, dtype=values.dtype
            )
            replicated = torch.scatter_add(
                out, 0, replica_ids, values.reshape(-1, replicas)
            )
            return replicated.sum(1)

        # This is the key trick to make these loss functions fast, as we can leverage single pass
        # kernels.
        intersection = segment_sum(sample_foreground_pred * sample_foreground_true)[1:]
        pred_sum = segment_sum(sample_foreground_pred)[1:]
        weighted_pred_sum = segment_sum(
            sample_foreground_pred * sample_foreground_pred.detach()
        )[1:]
        true_sum = segment_sum(sample_foreground_true)[1:]
        ce_sum = segment_sum(sample_ce_map)[1:]
        count = segment_sum(torch.ones_like(sample_foreground_pred))[1:]

        component_loss = _combine_vectorized_loss_terms(
            mode=mode,
            intersection=intersection,
            pred_sum=pred_sum,
            weighted_pred_sum=weighted_pred_sum,
            true_sum=true_sum,
            ce_sum=ce_sum,
            count=count,
            empty_cell_loss=empty_cell_loss,
            dice_smooth=dice_smooth,
        )

        valid_components = (count > 0).to(component_loss.dtype)
        cc_loss = (
            (component_loss * valid_components).sum()
            / valid_components.sum().clamp_min(1.0)
        )
        fallback = _whole_volume_fallback_loss(
            mode=mode,
            foreground_pred=sample_foreground_pred,
            foreground_true=sample_foreground_true,
            ce_map=sample_ce_map,
            smooth=dice_smooth,
        )
        return torch.where(valid_components.any(), cc_loss, fallback)

    per_sample = vmap(per_sample_loss, in_dims=(0, 0, 0, 0))(
        foreground_pred, foreground_true, ce_map, components
    )
    return per_sample.mean()


def _vectorized_blob_reduction(
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: torch.Tensor,
    components: torch.Tensor,
    mode: VectorizedLossMode,
    num_component_slots: int,
    dice_smooth: float = 0.0,
) -> torch.Tensor:
    num_voxels = foreground_pred[0].numel()
    replicas = _replica_count(num_voxels)

    def per_sample_loss(
        sample_foreground_pred: torch.Tensor,
        sample_foreground_true: torch.Tensor,
        sample_ce_map: torch.Tensor,
        sample_components: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_slots_with_bg = num_component_slots + 1
        # Each column accumulates into a separate replica.
        replica_ids = sample_components.reshape(-1, replicas)

        def segment_sum(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(
                num_slots_with_bg, replicas, device=values.device, dtype=values.dtype
            )
            replicated = torch.scatter_add(
                out, 0, replica_ids, values.reshape(-1, replicas)
            )
            return replicated.sum(1)

        intersection_bins = segment_sum(sample_foreground_pred * sample_foreground_true)
        pred_bins = segment_sum(sample_foreground_pred)
        true_bins = segment_sum(sample_foreground_true)
        ce_bins = segment_sum(sample_ce_map)
        count_bins = segment_sum(torch.ones_like(sample_foreground_pred))

        intersection = intersection_bins[0] + intersection_bins[1:]
        pred_sum = pred_bins[0] + pred_bins[1:]
        true_sum = true_bins[0] + true_bins[1:]
        ce_sum = ce_bins[0] + ce_bins[1:]
        # Blob loss averages masked CE over the whole patch. In practice this doesn't really matter,
        # as blob loss only masks other ground  truth, so the effective normalizer is basically constant anyways.
        count = ce_sum.new_full(ce_sum.shape, float(num_voxels))
        component_loss = _combine_vectorized_loss_terms(
            mode=mode,
            intersection=intersection,
            pred_sum=pred_sum,
            true_sum=true_sum,
            ce_sum=ce_sum,
            count=count,
            empty_cell_loss="dice",
            dice_smooth=dice_smooth,
        )

        valid_components = (count_bins[1:] > 0).to(component_loss.dtype)
        blob_loss = (
            (component_loss * valid_components).sum()
            / valid_components.sum().clamp_min(1.0)
        )
        has_components = valid_components.any()
        return blob_loss, has_components

    per_sample, sample_has_components = vmap(per_sample_loss, in_dims=(0, 0, 0, 0))(
        foreground_pred, foreground_true, ce_map, components
    )
    sample_has_components_f = sample_has_components.to(per_sample.dtype)
    return (
        (per_sample * sample_has_components_f).sum()
        / sample_has_components_f.sum().clamp_min(1.0)
    )
