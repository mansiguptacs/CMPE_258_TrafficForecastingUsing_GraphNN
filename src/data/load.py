import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tables


def load_speed(path: Path):
    """Read METR-LA speeds as (speeds[T, N] float32, timestamps, sensor_ids).

    The file was written by pandas 0.15 in HDF5 "fixed" format, which recent
    pandas cannot parse, so the arrays are read directly with PyTables.
    """
    with tables.open_file(str(path), mode="r") as f:
        g = f.root.df
        speeds = g.block0_values[:].astype(np.float32)
        timestamps = pd.to_datetime(g.axis1[:], unit="ns")
        sensor_ids = [s.decode() for s in g.axis0[:]]
    return speeds, timestamps, sensor_ids


def load_adjacency(path: Path):
    """Read the DCRNN road-distance adjacency as (sensor_ids, adj[N, N] float32)."""
    with open(path, "rb") as f:
        sensor_ids, _, adj = pickle.load(f, encoding="latin1")
    return [str(s) for s in sensor_ids], adj.astype(np.float32)
