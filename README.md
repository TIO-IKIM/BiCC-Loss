# BiCC Loss

BiCC is an instance-aware loss for binary 3D segmentation. It uses both annotated and
predicted connected components, so small lesions and false-positive predictions each
contribute to training. This repository contains the loss implementation and the
nnU-Net trainers used in the paper.

## Install

We recommend a CuPy-based version, as it is usually much faster.

```bash
pip install "bicc-loss[cuda12] @ git+https://github.com/TIO-IKIM/BiCC-Loss.git"  # CUDA 12 (CuPy)
pip install "bicc-loss[cuda13] @ git+https://github.com/TIO-IKIM/BiCC-Loss.git"  # CUDA 13 (CuPy)
pip install "bicc-loss @ git+https://github.com/TIO-IKIM/BiCC-Loss.git"          # CPU fallback (SciPy)
```

## Usage

BiCC can be used on its own or with nnU-Net (see below).

```python
from bicc_loss import VectorizedBiCCLoss

loss_fn = VectorizedBiCCLoss(alpha=0.5, sampling=(1.0, 0.5, 0.5), patch_size=(64, 128, 128))
instance_loss = loss_fn(logits, target)
```

- `logits`: `[B, 2, D, H, W]`, raw network output (softmax is applied inside).
- `target`: `[B, 1, D, H, W]`, `torch.int16`, values in `{0, 1}`.
- `sampling`: voxel spacing used for the Voronoi partition. `patch_size` lets the loss rescale
  `sampling` for lower-resolution deep supervision outputs.

We recommend combining `VectorizedBiCCLoss` with a global loss such as Dice + CE,
as this usually improves segmentation quality for instance-aware losses. Wrap it in
`torch.compile` for speed.
`VectorizedCCLoss` (reference partition only) and `VectorizedBlobLoss` are included as baselines.

## Paper training setup

`nnUNet/` contains nnU-Net at upstream commit
[`55124f6`](https://github.com/MIC-DKFZ/nnUNet/commit/55124f6524d775018160f2c7b341c87405c8c6a6)
plus the paper trainers in
`nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/instance/`.

```bash
git clone https://github.com/TIO-IKIM/BiCC-Loss.git && cd BiCC-Loss
pip install -e "nnUNet[cuda12]"   # or nnUNet[cuda13]; pulls bicc-loss from GitHub
nnUNetv2_train DATASET 3d_fullres FOLD -tr PaperBiCCDiceCE
```

| Trainer | Setting |
| --- | --- |
| `PaperDiceCE` | Dice + CE baseline |
| `PaperBlobDiceCE` | + blob loss |
| `PaperCCDiceCE` | + CC loss (α = 0) |
| `PaperBiCCDiceCE` | + BiCC loss (α = 0.5) |
| `PaperBiCCDiceCEAlpha025`, `…Alpha075`, `…Alpha1` | α ablation |
| `PaperBiCCDiceCESmoothedDice` | smoothed Dice instead of Eq. (3) for empty cells |

## License

Apache-2.0, as is nnU-Net.
