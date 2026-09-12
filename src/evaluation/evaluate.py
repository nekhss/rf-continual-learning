"""
Evaluation utilities for RF-CL.

Evaluation is strictly isolated from training and replay memory.

The evaluator:
    - sets the model to eval mode
    - uses torch.no_grad()
    - evaluates requested test tasks
    - restores the original model training state
    - never modifies replay memory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from .metrics import (
    accuracy,
    average_accuracy,
    average_forgetting,
    confusion_matrix,
    macro_f1,
    per_class_accuracy,
)


def evaluate_loader(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device | str = "cpu",
    num_classes: int = 11,
) -> Dict:
    """
    Evaluate one test DataLoader.

    Expected batches may be:

        {"x": ..., "y": ..., "snr": ..., "task": ...}

    or a tuple:

        (x, y)

    The evaluator does not use SNR/task as model inputs.
    """

    device = torch.device(device)

    was_training = model.training
    model.eval()

    all_targets: List[torch.Tensor] = []
    all_predictions: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for batch in loader:

                if isinstance(batch, Mapping):
                    x = batch["x"]
                    y = batch["y"]

                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]

                else:
                    raise TypeError(
                        "Unsupported evaluation batch format."
                    )

                x = x.to(device)
                y = torch.as_tensor(y, dtype=torch.long, device=device)

                logits = model(x)

                if logits.ndim != 2:
                    raise ValueError(
                        f"Expected logits [batch, classes], "
                        f"got {tuple(logits.shape)}."
                    )

                predictions = logits.argmax(dim=1)

                all_targets.append(y.detach().cpu())
                all_predictions.append(predictions.detach().cpu())

    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    if not all_targets:
        raise ValueError("Evaluation loader produced zero samples.")

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)

    cm = confusion_matrix(
        targets,
        predictions,
        num_classes=num_classes,
    )

    return {
        "accuracy": accuracy(targets, predictions),
        "macro_f1": macro_f1(
            targets,
            predictions,
            num_classes=num_classes,
        ),
        "per_class_accuracy": per_class_accuracy(
            targets,
            predictions,
            num_classes=num_classes,
        ),
        "confusion_matrix": cm.tolist(),
        "num_samples": int(targets.numel()),
    }


def evaluate_seen_tasks(
    model: torch.nn.Module,
    test_loaders: Mapping[int, Iterable],
    seen_tasks: Iterable[int],
    device: torch.device | str = "cpu",
    num_classes: int = 11,
) -> Dict[int, Dict]:
    """
    Evaluate only previously encountered test tasks.

    Future test sets are not touched.
    """

    results: Dict[int, Dict] = {}

    for task in sorted(seen_tasks):
        if task not in test_loaders:
            raise KeyError(
                f"No test loader supplied for task {task}."
            )

        results[task] = evaluate_loader(
            model=model,
            loader=test_loaders[task],
            device=device,
            num_classes=num_classes,
        )

    return results


def update_accuracy_matrix(
    accuracy_matrix: List[List[Optional[float]]],
    trained_through_task: int,
    task_results: Mapping[int, Dict],
    num_tasks: int = 5,
) -> List[List[Optional[float]]]:
    """
    Insert current evaluation results into the triangular matrix.

    Example:

        After T1:
            A11

        After T2:
            A21 A22

        ...

        After T5:
            A51 A52 A53 A54 A55
    """

    if not 1 <= trained_through_task <= num_tasks:
        raise ValueError(
            f"Invalid trained_through_task: {trained_through_task}"
        )

    if len(accuracy_matrix) != num_tasks:
        raise ValueError("Accuracy matrix has incorrect row count.")

    for row in accuracy_matrix:
        if len(row) != num_tasks:
            raise ValueError("Accuracy matrix has incorrect column count.")

    for task, result in task_results.items():
        if not 1 <= task <= trained_through_task:
            raise ValueError(
                "Evaluation contains a task that should not yet be visible."
            )

        if "accuracy" not in result:
            raise ValueError(
                f"Missing accuracy for task {task}."
            )

        accuracy_matrix[trained_through_task - 1][task - 1] = float(
            result["accuracy"]
        )

    return accuracy_matrix


def new_accuracy_matrix(
    num_tasks: int = 5,
) -> List[List[Optional[float]]]:
    """Create an empty triangular accuracy matrix."""

    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")

    return [
        [None for _ in range(num_tasks)]
        for _ in range(num_tasks)
    ]


def build_summary(
    accuracy_matrix: List[List[Optional[float]]],
) -> Dict[str, float]:
    """Calculate headline continual-learning metrics."""

    return {
        "average_accuracy": average_accuracy(
            accuracy_matrix
        ),
        "average_forgetting": average_forgetting(
            accuracy_matrix
        ),
    }


def save_results(
    results: Dict,
    path: str | Path,
) -> None:
    """Save experiment results as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            allow_nan=False,
        )