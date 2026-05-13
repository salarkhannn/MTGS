"""Reproducibility helpers."""

from __future__ import annotations

import random
from typing import Optional

from .config import SeedConfig


def _try_import_numpy() -> Optional[object]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


def _try_import_torch() -> Optional[object]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def set_seed(config: SeedConfig) -> None:
    random.seed(config.seed)

    np_mod = _try_import_numpy()
    if np_mod is not None:
        np_mod.random.seed(config.seed)

    torch_mod = _try_import_torch()
    if torch_mod is None:
        return

    torch_mod.manual_seed(config.seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(config.seed)

    torch_mod.backends.cudnn.deterministic = config.cudnn_deterministic
    torch_mod.backends.cudnn.benchmark = config.cudnn_benchmark

    if config.deterministic:
        torch_mod.use_deterministic_algorithms(True, warn_only=True)
