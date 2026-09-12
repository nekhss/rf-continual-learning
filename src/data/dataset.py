"""
Public PyTorch Dataset / DataLoader interface for RF-CL.

This is the ONLY module other developers (model / continual-learning /
training code) should need to import from:

    from src.data import get_task_dataset, get_task_dataloader

    train_loader = get_task_dataloader(1, "train")
    val_loader   = get_task_dataloader(1, "val")
    test_loader  = get_task_dataloader(1, "test")

    for batch in train_loader:
        x = batch["x"]     # [B, 2, 128] float32
        y = batch["y"]     # [B] int64, in [0, num_classes)
        snr = batch["snr"] # [B] int64, raw SNR in dB
        task = batch["task"]  # [B] int64, in {1,2,3,4}
        logits = model(x)

Internally this loads the raw dataset once per process (cached), builds
the deterministic master arrays, and indexes into them using the
pre-computed split indices in data/splits/splits_seed42.json. No
duplication of the full dataset beyond one master-array copy in memory.
"""

import functools
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.raw_loader import find_raw_dataset, load_raw_dict, load_master_arrays
from src.data.splits import load_splits

DEFAULT_DATA_ROOT = "data"
DEFAULT_SPLITS_PATH = os.path.join(DEFAULT_DATA_ROOT, "splits", "splits_seed42.json")

VALID_SPLITS = ("train", "val", "test")


class RFModDataset(Dataset):
    """A single (task, split) slice of RadioML2016.10a.

    Each item is a dict:
        {"x": FloatTensor[2, 128], "y": int, "snr": int, "task": int}
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        snr: np.ndarray,
        indices,
        task_id: int,
    ):
        self._X = X
        self._y = y
        self._snr = snr
        self._indices = np.asarray(indices, dtype=np.int64)
        self.task_id = int(task_id)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict:
        global_idx = self._indices[i]
        x = torch.from_numpy(self._X[global_idx]).float()  # [2, 128]
        return {
            "x": x,
            "y": int(self._y[global_idx]),
            "snr": int(self._snr[global_idx]),
            "task": self.task_id,
        }


@functools.lru_cache(maxsize=1)
def _load_cached_master(data_root: str):
    """Load + cache the raw dataset's master arrays for this process.
    lru_cache keyed on data_root so repeated calls (e.g. one per task/split
    requested) don't reload/reparse the raw pickle each time."""
    raw_path = find_raw_dataset(
        [
            os.path.join(data_root, "raw", "RML2016.10a_dict.pkl"),
            "/mnt/user-data/uploads/RML2016.10a_dict.pkl",
        ]
    )
    raw = load_raw_dict(raw_path)
    X, y, snr, class_names = load_master_arrays(raw)
    return X, y, snr, class_names


def _splits_path(data_root: str) -> str:
    return os.path.join(data_root, "splits", "splits_seed42.json")


def get_task_dataset(
    task_id: int,
    split: str,
    data_root: str = DEFAULT_DATA_ROOT,
) -> RFModDataset:
    """Return an RFModDataset for the given task (1-4) and split
    ('train' / 'val' / 'test'), using the saved deterministic split
    indices. Raises a clear error if prepare_data.py hasn't been run yet."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    splits_path = _splits_path(data_root)
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"No split file found at {splits_path}. Run "
            f"'python src/data/prepare_data.py' first to generate it."
        )

    splits = load_splits(splits_path)
    task_key = str(task_id)
    if task_key not in splits["tasks"]:
        raise ValueError(
            f"task_id={task_id} not found in splits file "
            f"(available: {list(splits['tasks'].keys())})"
        )

    indices = splits["tasks"][task_key][split]
    X, y, snr, _class_names = _load_cached_master(data_root)
    return RFModDataset(X, y, snr, indices, task_id=task_id)


def get_task_dataloader(
    task_id: int,
    split: str,
    batch_size: int = 128,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    data_root: str = DEFAULT_DATA_ROOT,
) -> DataLoader:
    """Return a DataLoader for the given task/split.

    shuffle defaults to True for 'train' and False otherwise, but can be
    overridden explicitly.
    """
    dataset = get_task_dataset(task_id, split, data_root=data_root)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
