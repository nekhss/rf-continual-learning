"""Synthetic data utilities -- DEV/SMOKE-TEST ONLY.

The real dataset (RadioML 2016.10a) is owned by Person 1 (``src/data/``). Until
that loader lands, these helpers let us exercise the model, trainer and both
baselines end-to-end.

Each sample matches the agreed contract exactly::

    {"x": FloatTensor[2, 128], "y": int, "snr": int, "task": int}

A faint per-class sinusoid is injected so the pipeline is *mildly* learnable
(accuracy climbs above chance), which makes smoke tests meaningful. Per-task
noise scale mimics "evolving SNR" -- but the model never receives snr/task.

Once ``src/data/`` is ready, drop these in favour of the real loader; the model
API (``logits = model(x)``) does not change.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

SIG_LEN = 128
IQ_CHANNELS = 2


class RFDictDataset(Dataset):
    """In-memory dataset yielding the agreed ``{x, y, snr, task}`` dict."""

    def __init__(self, x, y, snr, task):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.snr = torch.as_tensor(np.asarray(snr), dtype=torch.long)
        self.task = torch.as_tensor(np.asarray(task), dtype=torch.long)
        assert self.x.shape[1:] == (IQ_CHANNELS, SIG_LEN), self.x.shape

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        return {
            "x": self.x[idx],
            "y": int(self.y[idx]),
            "snr": int(self.snr[idx]),
            "task": int(self.task[idx]),
        }


def _make_split(n: int, num_classes: int, task_id: int, snr_db: int,
                rng: np.random.Generator) -> RFDictDataset:
    t = np.arange(SIG_LEN, dtype=np.float32) / SIG_LEN
    y = rng.integers(0, num_classes, size=n)
    noise_scale = 10.0 ** (-snr_db / 20.0)  # higher SNR -> less noise

    x = np.empty((n, IQ_CHANNELS, SIG_LEN), dtype=np.float32)
    for i in range(n):
        c = int(y[i])
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = (c + 1)
        i_ch = np.cos(2.0 * np.pi * freq * t + phase)
        q_ch = np.sin(2.0 * np.pi * freq * t + phase)
        x[i, 0] = i_ch + noise_scale * rng.standard_normal(SIG_LEN).astype(np.float32)
        x[i, 1] = q_ch + noise_scale * rng.standard_normal(SIG_LEN).astype(np.float32)

    snr = np.full(n, snr_db, dtype=np.int64)
    task = np.full(n, task_id, dtype=np.int64)
    return RFDictDataset(x, y, snr, task)


def make_synthetic_tasks(
    num_tasks: int = 5,
    num_classes: int = 11,
    train_per_task: int = 480,
    val_per_task: int = 120,
    test_per_task: int = 120,
    seed: int = 42,
    snr_schedule: List[int] | None = None,
) -> List[Dict[str, RFDictDataset]]:
    """Build ``num_tasks`` tasks, each with train/val/test splits.

    Returns a list of dicts ``{"train": ds, "val": ds, "test": ds}``.
    Tasks share the 11-class label space but differ in SNR (noise level),
    loosely mirroring the "evolving SNR conditions" setting.

    The default 5-task SNR schedule uses one representative SNR per band from
    the shared config (task_1 high SNR -> task_5 very low SNR). This is only a
    stand-in until Person 1's real loader builds tasks from ``config['tasks']``.
    """
    if snr_schedule is None:
        # Representative SNR per band: [12..18], [4..10], [-4..2], [-12..-6], [-20..-14]
        base = [15, 7, -1, -9, -17]
        if num_tasks <= len(base):
            snr_schedule = base[:num_tasks]
        else:
            snr_schedule = (base * ((num_tasks // len(base)) + 1))[:num_tasks]

    rng = np.random.default_rng(seed)
    tasks: List[Dict[str, RFDictDataset]] = []
    for t in range(num_tasks):
        snr_db = int(snr_schedule[t])
        tasks.append(
            {
                "train": _make_split(train_per_task, num_classes, t, snr_db, rng),
                "val": _make_split(val_per_task, num_classes, t, snr_db, rng),
                "test": _make_split(test_per_task, num_classes, t, snr_db, rng),
            }
        )
    return tasks
