"""STGCN runs, then a mean +/- std summary over seeds.

Graph study: distance, correlation and no-graph variants, three seeds each.
Imputation study: distance graph with other ways of filling missing inputs, seed 42.
Masked training: extra inputs hidden in every training batch, all three graph variants, three seeds.

    python -m src.models.stgcn.experiments run            # trains every run that has no metrics.json yet
    python -m src.models.stgcn.experiments summary        # writes results/stgcn_summary.json and prints the table
    python -m src.models.stgcn.experiments tune           # hyperparameter search runs, then the table below
    python -m src.models.stgcn.experiments tune-summary   # writes results/stgcn_tuning.json and prints it
"""
import argparse
import copy
import json

import numpy as np

from src.config import REPO_ROOT, load_config
from src.models.stgcn.train import train

SEEDS = (42, 43, 44)
MAX_EPOCHS = 80  # early stopping decides the real length; the cap only has to be high enough not to cut runs off

TRAIN_MASK = {"point_max": 0.5, "sensor_prob": 0.5, "sensor_max": 0.5}

# variant -> (adjacency, input_imputation, mask_channel, train_mask, seeds)
VARIANTS = {
    "distance": ("distance", "ffill", False, None, SEEDS),
    "correlation": ("correlation", "ffill", False, None, SEEDS),
    "no_graph": ("identity", "ffill", False, None, SEEDS),
    "distance_ffill_mask": ("distance", "ffill", True, None, (42,)),
    "distance_mean": ("distance", "mean", False, None, (42,)),
    "distance_mean_mask": ("distance", "mean", True, None, (42,)),
    "distance_mean_mask_aug": ("distance", "mean", True, TRAIN_MASK, SEEDS),
    "correlation_mean_mask_aug": ("correlation", "mean", True, TRAIN_MASK, SEEDS),
    "no_graph_mean_mask_aug": ("identity", "mean", True, TRAIN_MASK, SEEDS),
}

# Hyperparameter search: one change at a time from TUNE_BASE, seed 42, judged on validation MAE only.
TUNE_BASE = "distance_mean_mask_aug"
TUNING = {
    "dropout0.1": {"dropout": 0.1},
    "dropout0.5": {"dropout": 0.5},
    "wide": {"blocks": [[1, 64, 128], [128, 64, 256]]},
    "lr0.002": {"lr": 0.002},
    "lr0.0005": {"lr": 0.0005},
    "ks2": {"ks": 2},
    "ks4": {"ks": 4},
}

# stgcn_distance (seed 42) stopped early at epoch 38, so it is reused as-is. The Check-in 2
# stgcn_no_graph run hit its 40-epoch cap and is replaced by a converged run.
REUSED = {("distance", 42): "stgcn_distance"}


def tag_for(variant: str, seed: int) -> str:
    return REUSED.get((variant, seed), f"stgcn_{variant}_s{seed}")


def config_for(base: dict, variant: str, seed: int) -> dict:
    adjacency, imputation, mask_channel, train_mask, _ = VARIANTS[variant]
    cfg = copy.deepcopy(base)
    cfg["graph"]["adjacency"] = adjacency
    cfg["missing"]["input_imputation"] = imputation
    cfg["missing"]["mask_channel"] = mask_channel
    cfg["missing"]["train_mask"] = train_mask
    cfg["seed"] = seed
    return cfg


def runs() -> list[tuple[str, int, str]]:
    """Seed-major order, so a full first pass over all variants finishes before extra seeds."""
    return [(v, seed, tag_for(v, seed)) for seed in SEEDS for v, spec in VARIANTS.items() if seed in spec[-1]]


def run_all(limit: int | None = None) -> None:
    base = load_config()
    todo = [r for r in runs() if not (REPO_ROOT / "results" / r[2] / "metrics.json").exists()]
    for variant, seed, tag in todo[:limit]:
        train(config_for(base, variant, seed), tag=tag, epochs=MAX_EPOCHS)


def tune_tag(name: str) -> str:
    return f"stgcn_tune_{name}_s42"


def run_tuning(limit: int | None = None) -> None:
    base = load_config()
    todo = [n for n in TUNING if not (REPO_ROOT / "results" / tune_tag(n) / "metrics.json").exists()]
    for name in todo[:limit]:
        overrides = TUNING[name]
        cfg = config_for(base, TUNE_BASE, 42)
        cfg["stgcn"].update(overrides)
        train(cfg, tag=tune_tag(name), epochs=MAX_EPOCHS)


def tuning_table() -> list[dict]:
    """Validation MAE of every search run against the base setup; writes results/stgcn_tuning.json."""
    rows = []
    for name, tag in [("base", tag_for(TUNE_BASE, 42))] + [(n, tune_tag(n)) for n in TUNING]:
        path = REPO_ROOT / "results" / tag / "metrics.json"
        if not path.exists():
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"setting": name, "change": TUNING.get(name, {}), "val_mae": m["val"]["average"]["mae"],
                     "test_mae": m["test"]["average"]["mae"], "best_epoch": m["best_epoch"],
                     "parameters": m["parameters"], "train_minutes": m["train_minutes"]})
    with open(REPO_ROOT / "results" / "stgcn_tuning.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{'setting':12s} {'val MAE':>8s} {'test MAE':>9s} {'best ep':>8s} {'params':>9s} {'minutes':>8s}")
    for r in sorted(rows, key=lambda r: r["val_mae"]):
        print(f"{r['setting']:12s} {r['val_mae']:8.4f} {r['test_mae']:9.4f} {r['best_epoch']:8d} "
              f"{r['parameters']:9,d} {r['train_minutes']:8.1f}")
    return rows


def _mean_std(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def summarize() -> dict:
    summary = {}
    for variant, spec in VARIANTS.items():
        found = []
        for seed in spec[-1]:
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
    parser.add_argument("action", choices=["run", "summary", "tune", "tune-summary"])
    parser.add_argument("--limit", type=int, default=None, help="train at most this many unfinished runs")
    args = parser.parse_args()
    if args.action.startswith("tune"):
        if args.action == "tune":
            run_tuning(args.limit)
        tuning_table()
    else:
        if args.action == "run":
            run_all(args.limit)
        print_summary(summarize())
