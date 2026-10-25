import numpy as np


def observed_mask(speeds: np.ndarray, missing_value: float = 0.0) -> np.ndarray:
    """True where a reading is real. METR-LA marks sensor dropouts as 0."""
    return ~(np.isnan(speeds) | (speeds == missing_value))


def missing_stats(observed: np.ndarray) -> dict:
    missing = ~observed
    per_sensor = missing.mean(axis=0) * 100
    return {
        "overall_pct": float(missing.mean() * 100),
        "per_sensor_pct": per_sensor.tolist(),
        "worst_sensors": np.argsort(per_sensor)[::-1][:10].tolist(),
        "fully_missing_steps": int(missing.all(axis=1).sum()),
    }


def impute_ffill(speeds: np.ndarray, observed: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Carry the last observed reading forward per sensor; use `fallback` (e.g. train mean)
    before a sensor's first reading. Only past values are used, so nothing leaks from the future.
    """
    T, N = speeds.shape
    idx = np.where(observed, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = speeds[idx, np.arange(N)[None, :]]
    never_seen = ~np.maximum.accumulate(observed, axis=0)
    return np.where(never_seen, fallback[None, :], filled).astype(np.float32)


def apply_missing_mask(observed: np.ndarray, rate: float, seed: int = 0, mode: str = "point") -> np.ndarray:
    """Simulate extra sensor failures on top of the native gaps; returns a new observed mask.

    mode="point": each reading dropped independently with probability `rate`.
    mode="sensor": a `rate` fraction of sensors is dropped for the whole period.
    """
    rng = np.random.default_rng(seed)
    if mode == "point":
        drop = rng.random(observed.shape) < rate
    elif mode == "sensor":
        n = observed.shape[1]
        drop = np.zeros(observed.shape, dtype=bool)
        drop[:, rng.choice(n, size=round(n * rate), replace=False)] = True
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return observed & ~drop
