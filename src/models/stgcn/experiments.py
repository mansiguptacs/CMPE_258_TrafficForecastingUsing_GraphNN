"""Check-in 3 STGCN runs, then a mean +/- std summary over seeds.

Graph study: distance, correlation and no-graph variants, three seeds each.
Imputation study: distance graph with other ways of filling missing inputs, seed 42.

    python -m src.models.stgcn.experiments run       # trains every run that has no metrics.json yet
    python -m src.models.stgcn.experiments summary   # writes results/stgcn_summary.json and prints the table
"""
import argparse
import copy
import json

import numpy as np

from src.config import REPO_ROOT, load_config
from src.models.stgcn.train import train

SEEDS = (42, 43, 44)
MAX_EPOCHS = 80  # early stopping decides the real length; the cap only has to be high enough not to cut runs off

# variant -> (adjacency, input_imputation, mask_channel, seeds)
VARIANTS = {
    "distance": ("distance", "ffill", False, SEEDS),
    "correlation": ("correlation", "ffill", False, SEEDS),
    "no_graph": ("identity", "ffill", False, SEEDS),
    "distance_ffill_mask": ("distance", "ffill", True, (42,)),
    "distance_mean": ("distance", "mean", False, (42,)),
    "distance_mean_mask": ("distance", "mean", True, (42,)),
}

# stgcn_distance (seed 42) stopped early at epoch 38, so it is reused as-is. The Check-in 2
# stgcn_no_graph run hit its 40-epoch cap and is replaced by a converged run.
REUSED = {("distance", 42): "stgcn_distance"}


def tag_for(variant: str, seed: int) -> str:
    return REUSED.get((variant, seed), f"stgcn_{variant}_s{seed}")


def config_for(base: dict, variant: str, seed: int) -> dict:
    adjacency, imputation, mask_channel, _ = VARIANTS[variant]
    cfg = copy.deepcopy(base)
    cfg["graph"]["adjacency"] = adjacency
    cfg["missing"]["input_imputation"] = imputation
    cfg["missing"]["mask_channel"] = mask_channel
    cfg["seed"] = seed
    return cfg


def runs() -> list[tuple[str, int, str]]:
    """Seed-major order, so a full first pass over all variants finishes before extra seeds."""
    return [(v, seed, tag_for(v, seed)) for seed in SEEDS for v, spec in VARIANTS.items() if seed in spec[3]]


def run_all() -> None:
    base = load_config()
    for variant, seed, tag in runs():
        if (REPO_ROOT / "results" / tag / "metrics.json").exists():
            print(f"skip {tag}: already done", flush=True)
            continue
        train(config_for(base, variant, seed), tag=tag, epochs=MAX_EPOCHS)


def _mean_std(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def summarize() -> dict:
    summary = {}
    for variant, spec in VARIANTS.items():
        found = []
        for seed in spec[3]:
            path = REPO_ROOT / "results" / tag_for(variant, seed) / "metrics.json"
            if path.exists():
                with open(path) as f:
                    found.append((seed, json.load(f)))
        if not found:
            continue
        horizons = found[0][1]["test"].keys()
        summary[variant] = {
            "seeds": [s for s, _ in found],
            "best_epochs": [m["best_epoch"] for _, m in found],
            "test": {
                h: {k: _mean_std([m["test"][h][k] for _, m in found]) for k in ("mae", "rmse", "mape")}
                for h in horizons
            },
        }
    with open(REPO_ROOT / "results" / "stgcn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    print(f"{'variant':20s} {'seeds':>5s}  {'15min MAE':>13s}  {'30min MAE':>13s}  {'60min MAE':>13s}  {'avg MAE':>13s}")
    for variant, s in summary.items():
        cells = [f"{s['test'][h]['mae']['mean']:.3f} ± {s['test'][h]['mae']['std']:.3f}"
                 for h in ("15min", "30min", "60min", "average")]
        print(f"{variant:20s} {len(s['seeds']):>5d}  " + "  ".join(f"{c:>13s}" for c in cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGCN graph and imputation experiments")
    parser.add_argument("action", choices=["run", "summary"])
    args = parser.parse_args()
    if args.action == "run":
        run_all()
    print_summary(summarize())
