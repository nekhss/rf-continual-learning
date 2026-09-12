"""
Metrics for RF-CL continual-learning experiments.

Primary metrics
---------------
Average Accuracy:
    AA = mean(A_5j), j = 1..5

Average Forgetting:
    F1 = A11 - A51
    F2 = A22 - A52
    F3 = A33 - A53
    F4 = A44 - A54

    Average Forgetting = (F1 + F2 + F3 + F4) / 4

Task 5 is excluded from forgetting because there is no subsequent
task after Task 5.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


def accuracy(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> float:
    """Calculate classification accuracy."""

    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "targets and predictions must both be one-dimensional."
        )

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have identical shapes."
        )

    if targets.numel() == 0:
        raise ValueError("Cannot calculate accuracy on empty inputs.")

    return float((targets == predictions).float().mean().item())


def macro_f1(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = 11,
) -> float:
    """Calculate macro F1 without requiring sklearn."""

    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "targets and predictions must both be one-dimensional."
        )

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have identical shapes."
        )

    if targets.numel() == 0:
        raise ValueError("Cannot calculate F1 on empty inputs.")

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    f1_scores = []

    for class_id in range(num_classes):
        true_positive = (
            ((targets == class_id) & (predictions == class_id))
            .sum()
            .item()
        )

        false_positive = (
            ((targets != class_id) & (predictions == class_id))
            .sum()
            .item()
        )

        false_negative = (
            ((targets == class_id) & (predictions != class_id))
            .sum()
            .item()
        )

        denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )

        if denominator == 0:
            # Class absent from both target and prediction.
            # It contributes zero rather than producing NaN.
            f1 = 0.0
        else:
            f1 = (
                2.0 * true_positive
                / denominator
            )

        f1_scores.append(f1)

    return float(sum(f1_scores) / num_classes)


def per_class_accuracy(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = 11,
) -> list[float]:
    """Return accuracy for each modulation class."""

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have identical shapes."
        )

    if targets.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")

    result = []

    for class_id in range(num_classes):
        mask = targets == class_id

        count = int(mask.sum().item())

        if count == 0:
            result.append(float("nan"))
            continue

        class_acc = (
            predictions[mask] == targets[mask]
        ).float().mean().item()

        result.append(float(class_acc))

    return result


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int = 11,
) -> torch.Tensor:
    """Construct a multiclass confusion matrix."""

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have identical shapes."
        )

    if targets.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")

    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if not 0 <= target < num_classes:
            raise ValueError(f"Invalid target class: {target}")

        if not 0 <= prediction < num_classes:
            raise ValueError(f"Invalid prediction class: {prediction}")

        matrix[target, prediction] += 1

    return matrix


def average_accuracy(
    accuracy_matrix: Sequence[Sequence[Optional[float]]],
    final_task: int = 5,
) -> float:
    """
    Calculate final Average Accuracy.

    For five tasks:

        AA = (A51 + A52 + A53 + A54 + A55) / 5

    Only the final row is used.
    """

    if final_task <= 0:
        raise ValueError("final_task must be positive.")

    if len(accuracy_matrix) < final_task:
        raise ValueError(
            "Accuracy matrix does not contain the final task row."
        )

    final_row = accuracy_matrix[final_task - 1]

    if len(final_row) < final_task:
        raise ValueError(
            "Final accuracy row is incomplete."
        )

    values = final_row[:final_task]

    if any(value is None for value in values):
        raise ValueError(
            "Final accuracy row contains missing values."
        )

    return float(sum(values) / final_task)


def average_forgetting(
    accuracy_matrix: Sequence[Sequence[Optional[float]]],
) -> float:
    """
    Calculate Average Forgetting.

    For five tasks:

        F1 = A11 - A51
        F2 = A22 - A52
        F3 = A33 - A53
        F4 = A44 - A54

        AF = (F1 + F2 + F3 + F4) / 4

    Task 5 is excluded because no later task exists.

    The matrix must contain at least five rows and columns.
    """

    num_tasks = len(accuracy_matrix)

    if num_tasks < 5:
        raise ValueError(
            "Average Forgetting requires a five-task accuracy matrix."
        )

    for row in accuracy_matrix[:5]:
        if len(row) < 5:
            raise ValueError(
                "Average Forgetting requires five columns."
            )

    forgetting_values = []

    for task in range(4):
        initial = accuracy_matrix[task][task]
        final = accuracy_matrix[4][task]

        if initial is None or final is None:
            raise ValueError(
                f"Missing accuracy for task {task + 1}."
            )

        forgetting_values.append(initial - final)

    return float(sum(forgetting_values) / len(forgetting_values))