def chronological_split(num_steps: int, train: float, val: float, test: float) -> dict:
    """Return contiguous [start, end) index ranges over time: train, then val, then test.

    Windows are later cut inside each range, so no input or target crosses a split boundary.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")
    n_train = round(num_steps * train)
    n_val = round(num_steps * val)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, num_steps),
    }
