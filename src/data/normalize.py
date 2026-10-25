import json
from pathlib import Path

import numpy as np


class SensorScaler:
    """Per-sensor z-score fitted only on observed (non-missing) training readings."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def fit(self, speeds: np.ndarray, observed: np.ndarray) -> "SensorScaler":
        masked = np.where(observed, speeds, np.nan)
        self.mean = np.nanmean(masked, axis=0).astype(np.float32)
        std = np.nanstd(masked, axis=0).astype(np.float32)
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @classmethod
    def load(cls, path: Path) -> "SensorScaler":
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])
