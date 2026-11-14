"""Final comparison of every model on the same test windows, from each model's saved predictions.

Each run saves results/<tag>/test_predictions.npz with key "pred": test forecasts in mph,
shaped [6832, 12, 207] (windows in time order, all 12 steps, sensors in METR-LA column order).
Runs that also did the missing-data test save results/<tag>/missing_eval.json (src.eval.masked_inputs).
Models with no saved runs yet are skipped.

    python -m src.eval.compare     # writes results/comparison.json and prints the tables
"""
import json
import re

import numpy as np

from src.config import REPO_ROOT
from src.eval.masked_inputs import mask_settings, score, test_targets
from src.eval.metrics import horizon_metrics
from src.models.stgcn.experiments import SEEDS, tag_for

RESULTS = REPO_ROOT / "results"
CONGESTED_MPH = 40
HORIZONS = (3, 6, 12)


def _seed_runs(prefix: str) -> list[str]:
    """`prefix` itself and any `prefix_s<seed>` runs that have saved predictions."""
    pattern = re.compile(rf"{re.escape(prefix)}(_s\d+)?")
    return sorted(p.name for p in RESULTS.glob(f"{prefix}*")
                  if pattern.fullmatch(p.name) and (p / "test_predictions.npz").exists())


# display name -> run tags (one per seed). Historical average ignores recent readings, so its
# missing-data rows are its normal forecasts scored on every setting.
MODELS = {
    "Historical average": _seed_runs("ha"),
    "LSTM": _seed_runs("lstm"),
    "DCRNN, distance graph": _seed_runs("dcrnn_distance"),
    "DCRNN, correlation graph": _seed_runs("dcrnn_correlation"),
    "STGCN, distance graph": [tag_for("distance", s) for s in SEEDS],
    "STGCN, correlation graph": [tag_for("correlation", s) for s in SEEDS],
    "STGCN, no graph": [tag_for("no_graph", s) for s in SEEDS],
}
INPUT_FREE = {"Historical average"}
STGCN_VARIANT = {"STGCN, distance graph": "distance", "STGCN, correlation graph": "correlation",
                 "STGCN, no graph": "no_graph"}


def load_predictions(tag: str) -> np.ndarray | None:
    path = RESULTS / tag / "test_predictions.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["pred"].astype(np.float32)


def traffic_state_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    """MAE per horizon on congested (realised speed < 40 mph) and free-flow readings."""
    out = {}
    for h in HORIZONS:
        p, t, m = pred[:, h - 1], target[:, h - 1], mask[:, h - 1]
        congested = m & (t < CONGESTED_MPH)
        free = m & (t >= CONGESTED_MPH)
        out[f"{h * 5}min"] = {"congested": float(np.abs(p - t)[congested].mean()),
                              "free_flow": float(np.abs(p - t)[free].mean()),
                              "congested_share": float(congested.sum() / m.sum())}
    return out


def sensor_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """MAE per sensor over all forecast steps."""
    return (np.abs(pred - target) * mask).sum(axis=(0, 1)) / np.maximum(mask.sum(axis=(0, 1)), 1)


def _mean_std(vals: list[float]) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}


def _missing_rows(name: str, tag: str, pred: np.ndarray) -> list[dict]:
    if name in INPUT_FREE:
        return [score(pred, mode, rate, ms) for mode, rate, ms in mask_settings()]
    if name in STGCN_VARIANT:
        path = RESULTS / "missing_eval.json"
        if not path.exists():
            return []
        with open(path) as f:
            rows = json.load(f)
        seed = next(s for s in SEEDS if tag_for(STGCN_VARIANT[name], s) == tag)
        return [r for r in rows if r["variant"] == STGCN_VARIANT[name] and r["seed"] == seed]
    path = RESULTS / tag / "missing_eval.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _missing_summary(rows: list[dict]) -> list[dict]:
    """Average MAE (all steps) per setting, over mask draws and seeds."""
    out = []
    for mode, rate in dict.fromkeys((r["mode"], r["rate"]) for r in rows):
        sel = [r for r in rows if r["mode"] == mode and r["rate"] == rate]
        row = {"mode": mode, "rate": rate, "runs": len(sel),
               "mae_all": float(np.mean([r["all"]["average"]["mae"] for r in sel]))}
        hidden = [r["hidden_sensors"]["average"]["mae"] for r in sel if "hidden_sensors" in r]
        if hidden:
            row["mae_hidden_sensors"] = float(np.mean(hidden))
        out.append(row)
    return out


def compare() -> dict:
    target, mask = test_targets()
    summary = {}
    for name, tags in MODELS.items():
        found = [(tag, p) for tag in tags if (p := load_predictions(tag)) is not None]
        if not found:
            print(f"skip {name}: no saved predictions", flush=True)
            continue
        for tag, p in found:
            if p.shape != target.shape:
                raise ValueError(f"{tag}: predictions are {p.shape}, expected {target.shape}")
        metrics = [horizon_metrics(p, target, mask) for _, p in found]
        states = [traffic_state_mae(p, target, mask) for _, p in found]
        missing = [r for tag, p in found for r in _missing_rows(name, tag, p)]
        summary[name] = {
            "runs": [tag for tag, _ in found],
            "test": {h: {k: _mean_std([m[h][k] for m in metrics]) for k in ("mae", "rmse", "mape")}
                     for h in metrics[0]},
            "traffic_state": {h: {k: _mean_std([s[h][k] for s in states]) for k in states[0][h]}
                              for h in states[0]},
            "sensor_mae": np.mean([sensor_mae(p, target, mask) for _, p in found], axis=0).round(4).tolist(),
            "missing": _missing_summary(missing),
        }
    with open(RESULTS / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_comparison(summary: dict) -> None:
    def cell(d):
        return f"{d['mean']:.3f} ± {d['std']:.3f}" if d["std"] else f"{d['mean']:.3f}"

    print("\nTest MAE (mph), mean ± std over seeds")
    print(f"{'model':26s} {'runs':>4s}  " + "  ".join(f"{h:>13s}" for h in ("15min", "30min", "60min", "average")))
    for name, s in summary.items():
        print(f"{name:26s} {len(s['runs']):>4d}  "
              + "  ".join(f"{cell(s['test'][h]['mae']):>13s}" for h in ("15min", "30min", "60min", "average")))

    print(f"\nTest MAE (mph) by traffic state: congested (< {CONGESTED_MPH} mph) / free flow")
    print(f"{'model':26s}  " + "  ".join(f"{h:>13s}" for h in ("15min", "30min", "60min")))
    for name, s in summary.items():
        ts = s["traffic_state"]
        print(f"{name:26s}  " + "  ".join(
            f"{ts[h]['congested']['mean']:>6.2f} / {ts[h]['free_flow']['mean']:<4.2f}" for h in ("15min", "30min", "60min")))

    print("\nMissing inputs: average MAE (mph), all sensors | hidden sensors only")
    for name, s in summary.items():
        if not s["missing"]:
            print(f"{name:26s}  not run yet")
            continue
        cells = [f"{r['mode'][0]}{r['rate']:.1f}: {r['mae_all']:.2f}"
                 + (f"|{r['mae_hidden_sensors']:.2f}" if "mae_hidden_sensors" in r else "") for r in s["missing"]]
        print(f"{name:26s}  " + "  ".join(cells))


if __name__ == "__main__":
    print_comparison(compare())
