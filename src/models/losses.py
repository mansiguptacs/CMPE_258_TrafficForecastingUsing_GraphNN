import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over real readings only; missing targets contribute nothing."""
    mask = mask.float()
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1.0)
