"""Shared missing-data test for every model: the same hidden readings, the same windows, the same scoring.

Extra test readings are hidden on top of the native gaps (independent readings, or whole sensors),
the inputs are re-filled, and the targets stay the original real readings. Masks come from fixed
seeds, so every machine hides exactly the same readings.

Any model, in five lines (predictions in mph, shaped [windows, 12, sensors]):

    from src.eval.masked_inputs import mask_settings, masked_test_inputs, score, save_rows
    rows = []
    for mode, rate, mask_seed in mask_settings():
        x = masked_test_inputs(mode, rate, mask_seed)     # [6832, 12, 207], normalized, forward-filled
        rows.append(score(my_predict(x), mode, rate, mask_seed))
    save_rows("dcrnn_distance", rows)                     # results/dcrnn_distance/missing_eval.json

`masked_test_inputs` at rate 0 is exactly the normal test input, the same windows as
`processed["inputs"]` sliced in time order.
"""
import json
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import REPO_ROOT, load_config
from src.data.missing import apply_missing_mask
from src.data.pipeline import load_processed, model_inputs
from src.eval.metrics import horizon_metrics

RATES = {"point": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7), "sensor": (0.1, 0.2, 0.3, 0.5)}
MASK_SEEDS = (0, 1, 2)


@lru_cache(maxsize=1)
def _base() -> tuple[dict, dict]:
    cfg = load_config()
    return cfg, load_processed(cfg)


def mask_settings() -> list[tuple[str, float, int]]:
    """(mode, rate, mask_seed) for every test setting. Rate 0 hides nothing, so one draw is enough."""
    return [(mode, rate, ms) for mode, rates in RATES.items() for rate in rates
            for ms in (MASK_SEEDS if rate > 0 else MASK_SEEDS[:1])]


def corrupted_observed(data: dict, rate: float, mode: str, mask_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Observed mask with an extra `rate` of test readings hidden; also returns the fully hidden sensors."""
    a, b = data["meta"]["split_index"]["test"]
    observed = data["observed"].copy()
    test_obs = apply_missing_mask(observed[a:b], rate, seed=mask_seed, mode=mode)
    hidden = np.flatnonzero(~test_obs.any(axis=0) & observed[a:b].any(axis=0)) if mode == "sensor" else None
    observed[a:b] = test_obs
    return observed, hidden


def _windows(arr: np.ndarray, offset: int, length: int, count: int) -> np.ndarray:
    """[count, length, ...] windows starting at arr[offset], arr[offset + 1], ..."""
    view = sliding_window_view(arr[offset:offset + count + length - 1], length, axis=0)
    return np.ascontiguousarray(np.moveaxis(view, -1, 1))


def num_test_windows() -> int:
    cfg, data = _base()
    a, b = data["meta"]["split_index"]["test"]
    w = cfg["windows"]
    return (b - a) - w["input_len"] - w["output_len"] + 1


def masked_test_inputs(mode: str, rate: float, mask_seed: int = 0,
                       imputation: str = "ffill", mask_channel: bool = False) -> np.ndarray:
    """Normalized test inputs [windows, input_len, sensors] with the extra readings hidden and re-filled.

    imputation: 'ffill' (last real reading) or 'mean' (the sensor's train mean).
    mask_channel: adds a last axis of size 2 whose second channel is 1 for real readings, 0 for filled.
    """
    cfg, data = _base()
    cfg = {**cfg, "missing": {**cfg["missing"], "input_imputation": imputation, "mask_channel": mask_channel}}
    observed, _ = corrupted_observed(data, rate, mode, mask_seed)
    a, _ = data["meta"]["split_index"]["test"]
    inputs = model_inputs(cfg, data, observed)
    x = _windows(inputs, a, cfg["windows"]["input_len"], num_test_windows())
    return x if mask_channel else x[..., 0]


def test_targets() -> tuple[np.ndarray, np.ndarray]:
    """Test targets in mph and their real-reading mask, both [windows, output_len, sensors]."""
    cfg, data = _base()
    a, _ = data["meta"]["split_index"]["test"]
    w = cfg["windows"]
    start, n = a + w["input_len"], num_test_windows()
    return _windows(data["speeds"], start, w["output_len"], n), _windows(data["observed"], start, w["output_len"], n)


def score(pred: np.ndarray, mode: str, rate: float, mask_seed: int = 0) -> dict:
    """Metrics for predictions made from `masked_test_inputs(mode, rate, mask_seed)`.

    Always scored on the original real readings; sensor mode also scores the hidden sensors alone.
    """
    _, data = _base()
    target, mask = test_targets()
    if pred.shape != target.shape:
        raise ValueError(f"predictions must be {target.shape} in mph, got {pred.shape}")
    _, hidden = corrupted_observed(data, rate, mode, mask_seed)
    row = {"mode": mode, "rate": rate, "mask_seed": mask_seed, "all": horizon_metrics(pred, target, mask)}
    if hidden is not None and len(hidden):
        row["hidden_sensors"] = horizon_metrics(pred[..., hidden], target[..., hidden], mask[..., hidden])
    return row


def save_rows(tag: str, rows: list[dict]) -> None:
    out = REPO_ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "missing_eval.json", "w") as f:
        json.dump(rows, f, indent=2)
