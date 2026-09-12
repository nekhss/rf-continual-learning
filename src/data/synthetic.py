"""
Synthetic RadioML2016.10a-shaped fixture generator.

This is NOT part of the production pipeline and prepare_data.py never
imports it. It exists purely so that:

  1. tests/test_data.py can run deterministically without needing the real
     ~600MB RadioML2016.10a download, and
  2. this pipeline's logic can be demonstrated end-to-end in environments
     (like sandboxes) that cannot reach dataset mirrors.

It produces a pickle with the exact same {(mod, snr): array[N,2,128]}
structure as the real dataset, using the real 11 class names and the
real -20..18dB / 2dB-step SNR grid, just with a small, random per-group
example count so tests run fast.
"""

import pickle
from typing import Optional

import numpy as np

from src.data.tasks import EXPECTED_CLASS_NAMES, EXPECTED_SNR_VALUES


def make_synthetic_raw(
    examples_per_group: int = 40,
    seed: int = 0,
    class_names=None,
    snr_values=None,
):
    """Build a {(mod, snr): np.ndarray[n, 2, 128]} dict matching the real
    RadioML2016.10a structure (11 classes x 20 SNRs by default)."""
    rng = np.random.RandomState(seed)
    class_names = class_names or EXPECTED_CLASS_NAMES
    snr_values = snr_values or EXPECTED_SNR_VALUES

    raw = {}
    for mod in class_names:
        for snr in snr_values:
            raw[(mod, snr)] = rng.randn(examples_per_group, 2, 128).astype(np.float32)
    return raw


def write_synthetic_pickle(path: str, **kwargs) -> None:
    raw = make_synthetic_raw(**kwargs)
    with open(path, "wb") as f:
        pickle.dump(raw, f, protocol=2)
