from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
from scipy.ndimage import label as scipy_label

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt  # type: ignore
    from cupyx.scipy.ndimage import label as cp_label  # type: ignore
    _CUPY_IMPORT_ERROR = None
    _HAS_CUPY = True
except Exception as exc:  # pragma: no cover - CuPy required at runtime
    cp = None  # type: ignore
    cp_distance_transform_edt = None  # type: ignore
    cp_label = None  # type: ignore
    _CUPY_IMPORT_ERROR = exc
    _HAS_CUPY = False


def _torch_tensor_to_cupy(tensor: torch.Tensor) -> "cp.ndarray":
    if tensor.device.type != "cuda":
        raise ValueError("connected components expect CUDA tensors for CuPy execution.")
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bool:
        # CuPy cannot import boolean DLPack tensors.
        tensor = tensor.view(torch.uint8)
    return cp.from_dlpack(tensor)


def _ensure_cupy_array(array_like) -> "cp.ndarray":
    if isinstance(array_like, torch.Tensor):
        return _torch_tensor_to_cupy(array_like)

    if isinstance(array_like, cp.ndarray):
        return array_like

    return cp.from_dlpack(array_like)


def _cupy_array_to_torch(array: "cp.ndarray", *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_dlpack(array)

    if tensor.device != device:
        tensor = tensor.to(device=device)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _require_cupy() -> None:
    if not _HAS_CUPY:
        raise RuntimeError(
            "CuPy with CUDA support is required for CUDA inputs. Install bicc-loss[cuda]."
        ) from _CUPY_IMPORT_ERROR


def _normalize_sampling(
    sampling: Optional[Sequence[float]], spatial_ndim: int
) -> Optional[tuple[float, ...]]:
    if sampling is None:
        return None

    normalized = tuple(float(i) for i in sampling)
    if len(normalized) != spatial_ndim:
        raise ValueError(
            f"Expected sampling with {spatial_ndim} entries, got {len(normalized)}"
        )
    if any(i <= 0 for i in normalized):
        raise ValueError(f"Sampling values must be positive, got {normalized}")
    return normalized


def _component_output_shape(y: torch.Tensor, do_bg: bool) -> list[int]:
    assert y.ndim == 5
    out_shape = list(y.shape)
    if not do_bg and y.shape[1] != 1:
        out_shape[1] -= 1
    return out_shape


def _validate_label_output_dtype(output_dtype: torch.dtype) -> None:
    if output_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"Connected-component label maps require torch.int32 or torch.int64, got {output_dtype}"
        )


def _stack_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)
    return torch.stack(tensors, dim=0)


@torch.library.custom_op("bicc_loss::get_cc", mutates_args=())
def get_cc(
    y: torch.Tensor,
    do_bg: bool,
    output_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Assign per-channel connected-component ids.

    Args:
        y: Prediction or target tensor shaped ``[B, C, D, H, W]``. Single-channel
            inputs are treated as foreground-only masks.
        do_bg: Whether to keep channel 0 (usually background) in multi-channel
            assignments.
        output_dtype: Integer dtype for the returned component ids.
    """
    assert y.ndim == 5
    _validate_label_output_dtype(output_dtype)
    first_channel = 0 if do_bg or y.shape[1] == 1 else 1
    cc_assignments = []
    for batch_index in range(y.shape[0]):
        per_channel = []
        for channel_index in range(first_channel, y.shape[1]):
            mask = y[batch_index, channel_index] > 0
            if y.device.type == "cuda":
                _require_cupy()
                cc_per_channel = _cupy_array_to_torch(
                    connected_components_cupy(mask), device=y.device, dtype=output_dtype
                )
            else:
                cc_per_channel = connected_components_scipy(mask, dtype=output_dtype)
            per_channel.append(cc_per_channel)
        cc_assignments.append(_stack_tensors(per_channel))

    cc_assignments = _stack_tensors(cc_assignments).to(device=y.device)
    assert list(cc_assignments.shape) == _component_output_shape(y, do_bg)
    assert cc_assignments.dtype == output_dtype
    return cc_assignments


@get_cc.register_fake
def _(
    y: torch.Tensor,
    do_bg: bool,
    output_dtype: torch.dtype = torch.int64,
):
    return y.new_empty(_component_output_shape(y, do_bg), dtype=output_dtype)


@torch.library.custom_op("bicc_loss::get_voronoi", mutates_args=())
def get_voronoi(
    y: torch.Tensor,
    do_bg: bool,
    sampling: Optional[list[float]] = None,
    output_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Compute per-channel Voronoi maps around foreground components.

    Args:
        y: One-hot tensor with foreground channels, shaped ``[B, C, D, H, W]``.
        do_bg: Whether to keep the background channel in the output structure.
        sampling: Physical voxel spacing in spatial tensor order. When provided,
            Voronoi assignment uses this metric.
        output_dtype: Integer dtype for the returned component ids.
    """
    assert y.ndim == 5
    _validate_label_output_dtype(output_dtype)
    sampling = _normalize_sampling(sampling, y.ndim - 2)

    cc_assignments = []
    for batch_index in range(y.shape[0]):
        per_channel = []
        for channel_index in range(0 if do_bg else 1, y.shape[1]):
            mask = y[batch_index, channel_index] > 0
            if y.device.type == "cuda":
                _require_cupy()
                per_channel.append(
                    _cupy_array_to_torch(
                        compute_voronoi_cupy(mask, sampling=sampling),
                        device=y.device,
                        dtype=output_dtype,
                    )
                )
            else:
                per_channel.append(
                    compute_voronoi_scipy(mask, sampling=sampling, dtype=output_dtype)
                )
        cc_assignments.append(_stack_tensors(per_channel))

    cc_assignments = _stack_tensors(cc_assignments).to(device=y.device)
    expected_shape = list(y.shape)
    if not do_bg:
        expected_shape[1] -= 1
    assert list(cc_assignments.shape) == expected_shape
    assert cc_assignments.dtype == output_dtype
    return cc_assignments


@get_voronoi.register_fake
def _(
    y: torch.Tensor,
    do_bg: bool,
    sampling: Optional[list[float]] = None,
    output_dtype: torch.dtype = torch.int64,
):
    return y.new_empty(_component_output_shape(y, do_bg), dtype=output_dtype)


def _torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compute_voronoi_cupy(labels, sampling: Optional[Sequence[float]] = None):
    """Generate GPU Voronoi partitions for a binary mask via CuPy distance transforms.

    Args:
        labels: Boolean-like array on GPU representing the foreground mask. Accepts
            PyTorch tensors, CuPy arrays, or objects exposing ``__cuda_array_interface__``.
    """
    labeled_cc = connected_components_cupy(labels)
    indices = cp_distance_transform_edt(
        labeled_cc == 0,
        sampling=sampling,
        return_distances=False,
        return_indices=True,
        float64_distances=False,
    )
    return labeled_cc[tuple(indices)]


def connected_components_cupy(labels):
    """Label connected components on GPU using CuPy's ndimage utilities."""
    mask = _ensure_cupy_array(labels) > 0
    # Use full 26-connectivity in 3D.
    structure = cp.ones((3, 3, 3), dtype=bool)
    labeled_cc, _ = cp_label(mask, structure=structure)
    return labeled_cc


def _label_scipy(labels: torch.Tensor) -> np.ndarray:
    mask = _torch_tensor_to_numpy(labels > 0)
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled_cc, _ = scipy_label(mask, structure=structure)
    return labeled_cc


def compute_voronoi_scipy(
    labels: torch.Tensor,
    sampling: Optional[Sequence[float]] = None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Generate CPU Voronoi partitions for a binary mask via SciPy distance transforms."""
    labeled_cc = _label_scipy(labels)
    indices = scipy_distance_transform_edt(
        labeled_cc == 0,
        sampling=sampling,
        return_distances=False,
        return_indices=True,
    )
    voronoi = labeled_cc[tuple(indices)]
    return torch.from_numpy(voronoi).to(device=labels.device, dtype=dtype)


def connected_components_scipy(
    labels: torch.Tensor,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Label connected components on CPU using SciPy's ndimage utilities."""
    return torch.from_numpy(_label_scipy(labels)).to(device=labels.device, dtype=dtype)
