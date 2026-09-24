from nnunetv2.training.nnUNetTrainer.variants.loss.instance.paper_base import (
    PaperBase,
)


# Semantic baselines in Table 2. All use the same global DiceCE term.
class PaperDiceCE(PaperBase):
    instance_loss_kind = "none"


class PaperBlobDiceCE(PaperBase):
    instance_loss_kind = "blob"


class PaperCCDiceCE(PaperBase):
    """Reference-only endpoint (alpha=0), exactly CC-DiceCE."""

    instance_loss_kind = "cc"


class PaperBiCCDiceCE(PaperBase):
    instance_loss_kind = "bicc"


class PaperBiCCDiceCEAlpha025(PaperBiCCDiceCE):
    """Ablation weighting the prediction partition by alpha=1/4."""

    alpha = 0.25


class PaperBiCCDiceCEAlpha075(PaperBiCCDiceCE):
    """Ablation weighting the prediction partition by alpha=3/4."""

    alpha = 0.75


class PaperBiCCDiceCEAlpha1(PaperBiCCDiceCE):
    """Prediction-only partition endpoint (alpha=1)."""

    alpha = 1.0


class PaperBiCCDiceCESmoothedDice(PaperBiCCDiceCE):
    """Replace Eq. (3) with smoothed Dice in the prediction branch."""

    empty_cell_loss = "dice"
    pred_branch_dice_smooth = 1e-5
