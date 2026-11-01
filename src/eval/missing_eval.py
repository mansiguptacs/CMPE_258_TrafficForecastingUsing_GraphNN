"""Robustness to missing inputs: hide extra test readings, re-impute, and score the trained STGCNs.

Only the model inputs are corrupted. Targets and their masks stay the original real readings, so every
rate is scored on exactly the same values. Each model re-fills the hidden readings its own way
(forward-fill or mean, with or without the missing-indicator channel).

    python -m src.eval.missing_eval     # writes results/missing_eval.json
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import REPO_ROOT, load_config
from src.data.missing import apply_missing_mask
from src.data.pipeline import load_processed, model_inputs
from src.data.windows import TrafficWindowDataset
from src.eval.metrics import horizon_metrics
from src.models.stgcn.experiments import config_for, runs
from src.models.stgcn.train import build_model, pick_device, predict

RATES = {"point": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7), "sensor": (0.1, 0.2, 0.3, 0.5)}
MASK_SEEDS = (0, 1, 2)


def corrupted_observed(data: dict, rate: float, mode: str, mask_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Observed mask with an extra `rate` of test readings hidden; also returns the fully hidden sensors."""
    a, b = data["meta"]["split_index"]["test"]
    observed = data["observed"].copy()
    test_obs = apply_missing_mask(observed[a:b], rate, seed=mask_seed, mode=mode)
    hidden = np.flatnonzero(~test_obs.any(axis=0) & observed[a:b].any(axis=0)) if mode == "sensor" else None
    observed[a:b] = test_obs
    return observed, hidden


def corrupted_test_set(cfg: dict, data: dict, observed: np.ndarray) -> TrafficWindowDataset:
    a, b = data["meta"]["split_index"]["test"]
    w = cfg["windows"]
    inputs = model_inputs(cfg, data, observed)
    return TrafficWindowDataset(inputs, data["speeds"], data["observed"], a, b, w["input_len"], w["output_len"])


def evaluate() -> list[dict]:
    base = load_config()
    data = load_processed(base)
    device = pick_device()
    mean = torch.tensor(data["scaler"].mean, device=device)
    std = torch.tensor(data["scaler"].std, device=device)

    # Rate 0 hides nothing, so one draw is enough.
    masks = [(mode, rate, ms, *corrupted_observed(data, rate, mode, ms))
             for mode, rates in RATES.items() for rate in rates
             for ms in (MASK_SEEDS if rate > 0 else MASK_SEEDS[:1])]

    rows = []
    for variant, seed, tag in runs():
        ckpt = REPO_ROOT / "checkpoints" / tag / "best.pt"
        if not ckpt.exists():
            print(f"skip {tag}: no checkpoint", flush=True)
            continue
        cfg = config_for(base, variant, seed)
        model = build_model(cfg, data, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        for mode, rate, mask_seed, observed, hidden in masks:
            loader = DataLoader(corrupted_test_set(cfg, data, observed), batch_size=base["stgcn"]["batch_size"])
            pred, target, mask = predict(model, loader, mean, std, device)
            row = {"variant": variant, "seed": seed, "mode": mode, "rate": rate, "mask_seed": mask_seed,
                   "all": horizon_metrics(pred, target, mask)}
            if hidden is not None and len(hidden):
                row["hidden_sensors"] = horizon_metrics(pred[..., hidden], target[..., hidden], mask[..., hidden])
            rows.append(row)
            print(f"{tag:32s} {mode:6s} {rate:.1f} draw {mask_seed}  MAE {row['all']['average']['mae']:.3f}", flush=True)

    with open(REPO_ROOT / "results" / "missing_eval.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    evaluate()
