"""Robustness to missing inputs: hide extra test readings, re-impute, and score the trained STGCNs.

Only the model inputs are corrupted. Targets and their masks stay the original real readings, so every
rate is scored on exactly the same values.

    python -m src.eval.missing_eval     # writes results/missing_eval.json
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import REPO_ROOT, load_config
from src.data.missing import apply_missing_mask, impute_ffill
from src.data.pipeline import load_processed
from src.data.windows import TrafficWindowDataset
from src.eval.metrics import horizon_metrics
from src.models.stgcn.experiments import SEEDS, VARIANTS, tag_for
from src.models.stgcn.train import build_model, pick_device, predict

RATES = {"point": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7), "sensor": (0.1, 0.2, 0.3, 0.5)}
MASK_SEED = 0


def corrupted_test_set(cfg: dict, data: dict, rate: float, mode: str) -> tuple[TrafficWindowDataset, np.ndarray]:
    """Test windows whose inputs have an extra `rate` of readings hidden; also returns the hidden sensors."""
    a, b = data["meta"]["split_index"]["test"]
    observed = data["observed"].copy()
    test_obs = apply_missing_mask(observed[a:b], rate, seed=MASK_SEED, mode=mode)
    hidden = np.flatnonzero(~test_obs.any(axis=0) & observed[a:b].any(axis=0)) if mode == "sensor" else None
    observed[a:b] = test_obs
    scaler = data["scaler"]
    inputs = scaler.transform(impute_ffill(data["speeds"], observed, scaler.mean))
    w = cfg["windows"]
    ds = TrafficWindowDataset(inputs, data["speeds"], data["observed"], a, b, w["input_len"], w["output_len"])
    return ds, hidden


def evaluate() -> list[dict]:
    base = load_config()
    data = load_processed(base)
    device = pick_device()
    mean = torch.tensor(data["scaler"].mean, device=device)
    std = torch.tensor(data["scaler"].std, device=device)

    test_sets = [(mode, rate, *corrupted_test_set(base, data, rate, mode))
                 for mode, rates in RATES.items() for rate in rates]

    rows = []
    for variant, adjacency in VARIANTS:
        for seed in SEEDS:
            tag = tag_for(variant, seed)
            ckpt = REPO_ROOT / "checkpoints" / tag / "best.pt"
            if not ckpt.exists():
                print(f"skip {tag}: no checkpoint", flush=True)
                continue
            cfg = {**base, "graph": {**base["graph"], "adjacency": adjacency}}
            model = build_model(cfg, data, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            for mode, rate, ds, hidden in test_sets:
                loader = DataLoader(ds, batch_size=base["stgcn"]["batch_size"])
                pred, target, mask = predict(model, loader, mean, std, device)
                row = {"variant": variant, "seed": seed, "mode": mode, "rate": rate,
                       "all": horizon_metrics(pred, target, mask)}
                if hidden is not None and len(hidden):
                    row["hidden_sensors"] = horizon_metrics(pred[..., hidden], target[..., hidden], mask[..., hidden])
                rows.append(row)
                print(f"{tag:24s} {mode:6s} {rate:.1f}  MAE {row['all']['average']['mae']:.3f}", flush=True)

    with open(REPO_ROOT / "results" / "missing_eval.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    evaluate()
