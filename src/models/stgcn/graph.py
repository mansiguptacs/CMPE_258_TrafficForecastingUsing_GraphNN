import numpy as np

from src.config import processed_dir


def get_adjacency(cfg: dict, data: dict) -> np.ndarray:
    """Adjacency selected by cfg['graph']['adjacency']: 'distance' (bundled with METR-LA)
    or 'correlation' (built from the training split and saved as processed/adj_correlation.npy).
    """
    kind = cfg["graph"]["adjacency"]
    if kind == "distance":
        return data["adj_distance"]
    if kind == "correlation":
        path = processed_dir(cfg) / "adj_correlation.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build the correlation graph first")
        return np.load(path)
    raise ValueError(f"unknown adjacency {kind!r}")


def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """L~ = 2 L / lambda_max - I with L = I - D^-1/2 W D^-1/2, as used by Chebyshev graph convolution.

    STGCN assumes an undirected graph, so the directed road adjacency is symmetrized with max(W, W^T).
    """
    w = np.maximum(adj, adj.T).astype(np.float64)
    np.fill_diagonal(w, 0.0)
    d = w.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    n = w.shape[0]
    lap = np.eye(n) - d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :]
    lambda_max = np.linalg.eigvalsh(lap).max()
    return (2.0 * lap / lambda_max - np.eye(n)).astype(np.float32)
