"""Baseline #1 -- Naive Sequential Fine-Tuning.

Train Task 1 -> Task 2 -> Task 3 -> Task 4 on ONE model, in order.

Rules enforced here:
  * When learning task N, ONLY task N's TRAIN data is used.
  * No replay of previous samples, no peeking at future tasks.
  * Test data is used ONLY for reporting (never for training or stopping).
  * Validation data may drive early stopping / hyperparameters.

This is NOT a continual-learning method -- it is the lower-bound reference that
exhibits catastrophic forgetting. Replay/EWC/LwF etc. live in Person 3's files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training.trainer import accuracy, fit, get_device


def run_naive_sequential(
    model: nn.Module,
    tasks: List[Dict[str, Optional[DataLoader]]],
    config: Optional[dict] = None,
    device=None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the naive sequential baseline.

    Parameters
    ----------
    model:
        A freshly-initialised classifier. The SAME model is fine-tuned across
        all tasks (that is the whole point of the baseline).
    tasks:
        Ordered list, one entry per task::

            {"train": DataLoader, "val": DataLoader | None, "test": DataLoader}

        Loaders (not raw datasets) are expected so that the exact same splits
        are shared across every baseline.
    config, device, verbose:
        Passed through to the generic trainer.

    Returns
    -------
    dict with keys:
        ``accuracy_matrix``      : (T x T) numpy array, lower-triangular, NaN
                                   above the diagonal. Entry [i, j] = accuracy on
                                   test set j after training through task i.
        ``accuracy_matrix_list`` : same matrix as nested lists (JSON friendly;
                                   NaN -> None).
        ``histories``            : per-task training history from ``fit``.
        ``final_accuracies``     : accuracy on each test set after the last task.
        ``num_tasks``            : T.

    The matrix is exposed in full so the evaluation module (Person 3) can derive
    average accuracy, forgetting / backward transfer, etc., without re-running.
    """
    device = device or get_device()
    model.to(device)

    num_tasks = len(tasks)
    if num_tasks == 0:
        raise ValueError("`tasks` must contain at least one task")

    matrix = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    histories: List[dict] = []

    for i, task in enumerate(tasks):
        train_loader = task.get("train")
        val_loader = task.get("val")
        if train_loader is None:
            raise ValueError(f"task {i} is missing a 'train' loader")

        if verbose:
            print(f"\n=== [naive] training task {i + 1}/{num_tasks} "
                  f"(train only, no replay) ===", flush=True)

        # ---- Train on CURRENT task's TRAIN split only -----------------------
        history = fit(
            model, train_loader, val_loader,
            config=config, device=device, verbose=verbose,
            log_prefix=f"[naive T{i + 1}] ",
        )
        histories.append(history)

        # ---- Evaluate on every test set seen SO FAR (0..i) ------------------
        for j in range(i + 1):
            test_loader = tasks[j].get("test")
            if test_loader is None:
                raise ValueError(f"task {j} is missing a 'test' loader")
            acc = accuracy(model, test_loader, device)
            matrix[i, j] = acc
            if verbose:
                print(f"[naive] after T{i + 1}: acc on Test T{j + 1} = {acc:.4f}",
                      flush=True)

    final_accuracies = matrix[num_tasks - 1, :num_tasks].tolist()

    return {
        "method": "naive_sequential",
        "num_tasks": num_tasks,
        "accuracy_matrix": matrix,
        "accuracy_matrix_list": _matrix_to_jsonable(matrix),
        "final_accuracies": final_accuracies,
        "histories": histories,
    }


def _matrix_to_jsonable(matrix: np.ndarray) -> List[List[Optional[float]]]:
    """Convert an ndarray with NaNs into nested lists (NaN -> None)."""
    out: List[List[Optional[float]]] = []
    for row in matrix:
        out.append([None if np.isnan(v) else float(v) for v in row])
    return out
