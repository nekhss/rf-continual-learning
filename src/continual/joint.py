"""Baseline #2 -- Joint (multi-task) Training.

Combine the TRAIN data of ALL tasks and train a single model once. This is the
classic upper-bound reference: it shows what accuracy is reachable when there is
no sequential constraint and no forgetting.

Rules enforced here:
  * Training uses the union of all tasks' TRAIN splits ONLY.
  * Test data is never used for training.
  * Validation data is not trained on; it may be used (concatenated) purely for
    early-stopping / monitoring.

Joint training is NOT a continual-learning method -- it violates the sequential
setting on purpose to give an upper bound.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ..training.trainer import accuracy, build_loader, fit, get_device


def _concat_datasets_from_loaders(loaders: List[Optional[DataLoader]]):
    datasets = [ld.dataset for ld in loaders if ld is not None]
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def run_joint_training(
    model: nn.Module,
    tasks: List[Dict[str, Optional[DataLoader]]],
    config: Optional[dict] = None,
    device=None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the joint-training upper-bound baseline.

    Parameters
    ----------
    model:
        A freshly-initialised classifier.
    tasks:
        Same structure as :func:`run_naive_sequential` -- a list of
        ``{"train", "val", "test"}`` DataLoaders. Train/val datasets are pulled
        from the loaders and concatenated; test loaders are evaluated per task.

    Returns
    -------
    dict with keys:
        ``per_task_accuracy`` : list, test accuracy on each task after joint
                                training (index j -> Test T(j+1)).
        ``mean_accuracy``     : mean of ``per_task_accuracy``.
        ``history``           : training history from ``fit``.
        ``num_tasks``         : T.
    """
    device = device or get_device()
    model.to(device)

    num_tasks = len(tasks)
    if num_tasks == 0:
        raise ValueError("`tasks` must contain at least one task")

    # ---- Build combined loaders (TRAIN union; VAL union for monitoring) -----
    train_ds = _concat_datasets_from_loaders([t.get("train") for t in tasks])
    if train_ds is None:
        raise ValueError("no training data found across tasks")
    val_ds = _concat_datasets_from_loaders([t.get("val") for t in tasks])

    train_loader = build_loader(train_ds, config, shuffle=True, seed=seed)
    val_loader = build_loader(val_ds, config, shuffle=False, seed=seed) if val_ds else None

    if verbose:
        print(f"\n=== [joint] training on union of {num_tasks} tasks' TRAIN data "
              f"({len(train_ds)} samples) ===", flush=True)

    history = fit(
        model, train_loader, val_loader,
        config=config, device=device, verbose=verbose, log_prefix="[joint] ",
    )

    # ---- Evaluate on each task's TEST set -----------------------------------
    per_task_accuracy: List[float] = []
    for j, task in enumerate(tasks):
        test_loader = task.get("test")
        if test_loader is None:
            raise ValueError(f"task {j} is missing a 'test' loader")
        acc = accuracy(model, test_loader, device)
        per_task_accuracy.append(acc)
        if verbose:
            print(f"[joint] acc on Test T{j + 1} = {acc:.4f}", flush=True)

    mean_acc = float(np.mean(per_task_accuracy)) if per_task_accuracy else float("nan")

    return {
        "method": "joint_training",
        "num_tasks": num_tasks,
        "per_task_accuracy": per_task_accuracy,
        "mean_accuracy": mean_acc,
        "history": history,
    }
