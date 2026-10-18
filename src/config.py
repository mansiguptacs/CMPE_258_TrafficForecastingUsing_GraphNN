import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "metr_la.yaml"


def is_colab() -> bool:
    return "google.colab" in sys.modules


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def data_root(cfg: dict) -> Path:
    """$TRAFFIC_DATA_ROOT wins, then the Drive folder on Colab, then <repo>/data."""
    env = os.environ.get("TRAFFIC_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    if is_colab():
        return Path(cfg["paths"]["colab_root"])
    local = Path(cfg["paths"]["local_root"])
    return local if local.is_absolute() else REPO_ROOT / local


def raw_dir(cfg: dict) -> Path:
    return data_root(cfg) / cfg["paths"]["raw_dir"]


def processed_dir(cfg: dict) -> Path:
    return data_root(cfg) / cfg["paths"]["processed_dir"]
