"""Central lookup for weights and datasets, which live in checkpoints/ and data/ at the repo root."""
import re
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

PATHS = SimpleNamespace(
    repo=str(REPO_ROOT),
    checkpoints=str(REPO_ROOT / "checkpoints"),
    weights=str(REPO_ROOT / "checkpoints" / "policy"),
    nwm_logs=str(REPO_ROOT / "checkpoints" / "world_model"),
    data_root=str(REPO_ROOT / "data"),
    data_splits=str(REPO_ROOT / "train" / "vint_train" / "data" / "data_splits"),
)


def expand_paths(config):
    """Recursively replace ${name} in a loaded YAML config with values from PATHS."""
    if isinstance(config, dict):
        return {key: expand_paths(value) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_paths(value) for value in config]
    if isinstance(config, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: getattr(PATHS, m.group(1)), config)
    return config
