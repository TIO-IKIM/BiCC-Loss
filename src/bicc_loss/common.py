from typing import Literal, Optional, Sequence

import torch


VectorizedLossMode = Literal["dice", "ce", "dice_ce"]

# Dice uses the standard smoothed loss for empty cells.
# Detached uses the paper's stop-gradient formulation.
EmptyCellLoss = Literal["dice", "detached"]


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def configure_instance_loss_dynamo() -> None:
    if not hasattr(torch, "_dynamo"):
        return
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 128
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 128
    if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = True


def _normalize_sampling(
    sampling: Optional[Sequence[float]],
) -> Optional[tuple[float, ...]]:
    if sampling is None:
        return None
    normalized = tuple(float(i) for i in sampling)
    if any(i <= 0 for i in normalized):
        raise ValueError(f"sampling values must be positive, got {normalized}")
    return normalized


def _normalize_patch_size(
    patch_size: Optional[Sequence[int]],
) -> Optional[tuple[int, ...]]:
    if patch_size is None:
        return None
    normalized = tuple(int(i) for i in patch_size)
    if any(i <= 0 for i in normalized):
        raise ValueError(f"patch_size values must be positive, got {normalized}")
    return normalized
