"""Forecast metrics shared by every model.

All metrics use observed readings only: wherever `mask` is False the target was a
sensor dropout, so it counts for nothing. Predictions and targets are in mph.
"""
import numpy as np


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """MAE, RMSE (mph) and MAPE (%) over the observed entries of any-shaped arrays."""
    mask = mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    err = pred[mask] - target[mask]
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((np.abs(err) / target[mask]).mean() * 100),
        "n": n,
    }


def horizon_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                    horizons=(3, 6, 12), interval_minutes: int = 5) -> dict:
    """Metrics per forecast horizon for arrays shaped [windows, steps, sensors].

    Horizon h is step h of the forecast, i.e. h * interval_minutes minutes ahead.
    The key "average" covers all forecast steps together.
    """
    out = {f"{h * interval_minutes}min": masked_metrics(pred[:, h - 1], target[:, h - 1], mask[:, h - 1])
           for h in horizons}
    out["average"] = masked_metrics(pred, target, mask)
    return out
