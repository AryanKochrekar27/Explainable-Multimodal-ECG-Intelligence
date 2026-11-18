"""
Configuration and Reproducibility Utilities
Loads configs/training_config.yaml and enforces deterministic random seeds.
"""
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


DEFAULT_CONFIG_PATH = Path("configs/training_config.yaml")


def load_training_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load training hyperparameters and paths from YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration in {path}")
    return config


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

