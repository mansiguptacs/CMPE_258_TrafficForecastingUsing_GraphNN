import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, processed_dir, raw_dir
from src.data.load import load_adjacency, load_speed
from src.data.missing import impute_ffill, missing_stats, observed_mask
from src.data.normalize import SensorScaler
from src.data.split import chronological_split
from src.data.windows import TrafficWindowDataset

PROCESSED_FILE = "metr_la_processed.npz"
SCALER_FILE = "scalers.json"
META_FILE = "meta.json"


def prepare(cfg: dict) -> dict:
    """Raw METR-LA -> processed arrays, scaler and metadata shared by every model."""
    src, out = raw_dir(cfg), processed_dir(cfg)
    speeds, timestamps, sensor_ids = load_speed(src / cfg["paths"]["speed_file"])
    adj_ids, adj = load_adjacency(src / cfg["paths"]["adjacency_file"])
    if adj_ids != sensor_ids:
        raise ValueError("adjacency sensor order does not match speed columns")

    observed = observed_mask(speeds, cfg["normalization"]["missing_value"])
    s = cfg["split"]
    ranges = chronological_split(len(speeds), s["train"], s["val"], s["test"])
    tr0, tr1 = ranges["train"]

    scaler = SensorScaler().fit(speeds[tr0:tr1], observed[tr0:tr1])
    inputs = scaler.transform(impute_ffill(speeds, observed, scaler.mean))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / PROCESSED_FILE,
        speeds=speeds,
        inputs=inputs,
        observed=observed,
        timestamps=timestamps.asi8,
        sensor_ids=np.array(sensor_ids),
        adj_distance=adj,
    )
    scaler.save(out / SCALER_FILE)

    stats = missing_stats(observed)
    meta = {
        "shape": {"timesteps": len(speeds), "sensors": len(sensor_ids)},
        "time_range": [str(timestamps[0]), str(timestamps[-1])],
        "split_index": ranges,
        "split_time": {
            k: [str(timestamps[a]), str(timestamps[b - 1])] for k, (a, b) in ranges.items()
        },
        "missing_pct_overall": stats["overall_pct"],
        "missing_pct_by_split": {
            k: float((~observed[a:b]).mean() * 100) for k, (a, b) in ranges.items()
        },
        "fully_missing_steps": stats["fully_missing_steps"],
        "input_imputation": cfg["missing"]["input_imputation"],
        "windows": cfg["windows"],
    }
    with open(out / META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_processed(cfg: dict) -> dict:
    out = processed_dir(cfg)
    with np.load(out / PROCESSED_FILE) as z:
        data = {k: z[k] for k in z.files}
    data["timestamps"] = pd.to_datetime(data["timestamps"], unit="ns")
    data["sensor_ids"] = data["sensor_ids"].tolist()
    data["scaler"] = SensorScaler.load(out / SCALER_FILE)
    with open(out / META_FILE) as f:
        data["meta"] = json.load(f)
    return data


def make_datasets(cfg: dict, data: dict | None = None) -> dict:
    data = data or load_processed(cfg)
    w = cfg["windows"]
    return {
        split: TrafficWindowDataset(
            data["inputs"], data["speeds"], data["observed"], a, b, w["input_len"], w["output_len"]
        )
        for split, (a, b) in data["meta"]["split_index"].items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess METR-LA")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))
