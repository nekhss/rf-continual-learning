"""Centralized reproducibility helpers.

Seed logic lives here ONCE so no module re-implements it. Call
:func:`set_seed` at the start of every experiment/test.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, PyTorch (CPU + CUDA if present).

    Parameters
    ----------
    seed:
        The RNG seed. Project default is 42.
    deterministic:
        If True, request deterministic cuDNN/algorithms where practical.
        Uses ``warn_only=True`` so ops without a deterministic implementation
        degrade gracefully instead of raising.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch may not support warn_only; determinism is best-effort.
            pass
    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - only used with num_workers>0
    """DataLoader ``worker_init_fn`` for reproducible multi-worker loading."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Return a torch.Generator seeded for deterministic DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
