"""
config.py — shared YAML configuration loader for RolloutSecBench training scripts.

Every script under training/ resolves configs/*.yaml through this module instead of
computing its own os.path.join(BASE_DIR, ...) logic or hardcoding values inline. That
duplication is exactly how paths and hyperparameters drift out of sync between scripts —
this module exists so there is exactly one place that knows where the repo root and
configs/ directory are.
"""

import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


def load_config(name: str) -> dict:
    """
    Load configs/<name>.yaml (name without extension) and return it as a dict.
    Raises FileNotFoundError with a pointed message rather than a generic one, since a
    missing config file here means someone hardcoded a value back into a script instead of
    using this loader.
    """
    path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. All experiment configuration lives in "
            f"configs/ (see configs/README.md) — do not hardcode values back into scripts."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Resolve a repo-root-relative path (as stored in configs/paths.yaml) to an absolute one."""
    return os.path.join(REPO_ROOT, relative_path)
