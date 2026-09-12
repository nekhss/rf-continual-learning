"""
Deterministic, stratified train/val/test splitting for RF-CL.

Splits are computed *per task* (each task only ever sees its own SNR
range's examples), stratified by modulation class, with a fixed seed.
The resulting global example indices (indices into the master arrays
built by ``raw_loader.load_master_arrays``) are persisted to
``data/splits/splits_seed42.json`` so every experiment -- baseline,
naive continual, replay, ablations -- trains/validates/tests on the
exact same examples.

Because task membership is determined solely by SNR, and every example
has exactly one SNR, no example can ever belong to more than one task.
Because splitting happens independently per task on disjoint index sets,
no example can appear in more than one of {train, val, test} for its task.
"""

import json
import os
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.tasks import TASK_SNR_RANGES

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9


def build_task_splits(
    y: np.ndarray, snr: np.ndarray, seed: int = SEED
) -> Dict[int, Dict[str, List[int]]]:
    """Compute stratified train/val/test index lists for each task.

    Args:
        y:   int64 array [N], class label per example (global master-array index space)
        snr: int64 array [N], SNR per example (same index space)
        seed: random seed for the split (default 42)

    Returns:
        {task_id: {"train": [...], "val": [...], "test": [...]}}
        All index lists are lists of Python ints, indices into the
        original master arrays (X, y, snr).
    """
    result: Dict[int, Dict[str, List[int]]] = {}

    for task_id, snr_list in TASK_SNR_RANGES.items():
        task_mask = np.isin(snr, snr_list)
        task_indices = np.nonzero(task_mask)[0]
        task_y = y[task_indices]

        if len(task_indices) == 0:
            raise ValueError(
                f"Task {task_id} (SNR={snr_list}) has zero matching examples "
                f"in the dataset -- cannot build a split for it."
            )

        # Step 1: carve off test (15%), stratified by class.
        train_val_idx, test_idx, train_val_y, _ = train_test_split(
            task_indices,
            task_y,
            test_size=TEST_FRAC,
            random_state=seed,
            stratify=task_y,
        )

        # Step 2: split remaining 85% into train (70% of total) and
        # val (15% of total) -> val is 15/85 of the remaining pool.
        val_relative_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative_frac,
            random_state=seed,
            stratify=train_val_y,
        )

        result[task_id] = {
            "train": sorted(int(i) for i in train_idx),
            "val": sorted(int(i) for i in val_idx),
            "test": sorted(int(i) for i in test_idx),
        }

    return result


def save_splits(
    splits: Dict[int, Dict[str, List[int]]],
    class_names: List[str],
    path: str,
    seed: int = SEED,
) -> None:
    payload = {
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "class_names": class_names,
        "task_snr_ranges": TASK_SNR_RANGES,
        "index_space": (
            "Indices refer to positions in the master arrays produced by "
            "src.data.raw_loader.load_master_arrays() when run against the "
            "same raw dataset file. That ordering is deterministic "
            "(sorted by class name, then SNR, then original array order), "
            "so these indices are reproducible without needing to persist "
            "the arrays themselves."
        ),
        "tasks": {
            str(task_id): {
                "snr_range": TASK_SNR_RANGES[task_id],
                "train": task_splits["train"],
                "val": task_splits["val"],
                "test": task_splits["test"],
            }
            for task_id, task_splits in splits.items()
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def validate_splits(splits: Dict[int, Dict[str, List[int]]]) -> List[str]:
    """Run the integrity checks required by the spec. Returns a list of
    problem descriptions (empty = all checks passed)."""
    problems = []

    all_task_indices = {}
    for task_id, s in splits.items():
        train, val, test = set(s["train"]), set(s["val"]), set(s["test"])

        if train & val:
            problems.append(f"Task {task_id}: train/val overlap ({len(train & val)} examples)")
        if train & test:
            problems.append(f"Task {task_id}: train/test overlap ({len(train & test)} examples)")
        if val & test:
            problems.append(f"Task {task_id}: val/test overlap ({len(val & test)} examples)")

        all_task_indices[task_id] = train | val | test

    # No example should belong to more than one task.
    seen = set()
    for task_id, idx_set in all_task_indices.items():
        overlap = seen & idx_set
        if overlap:
            problems.append(
                f"Task {task_id} shares {len(overlap)} example indices with "
                f"an earlier task -- an example belongs to more than one task."
            )
        seen |= idx_set

    return problems
