import math

import pytest
import torch

from src.evaluation.metrics import (
    accuracy,
    average_accuracy,
    average_forgetting,
    confusion_matrix,
    macro_f1,
    per_class_accuracy,
)


def test_accuracy():
    targets = torch.tensor([0, 1, 2, 1])
    predictions = torch.tensor([0, 1, 0, 1])

    assert accuracy(targets, predictions) == pytest.approx(0.75)


def test_accuracy_empty_rejected():
    targets = torch.tensor([], dtype=torch.long)
    predictions = torch.tensor([], dtype=torch.long)

    with pytest.raises(ValueError):
        accuracy(targets, predictions)


def test_macro_f1():
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])

    # Class 0:
    # TP=1, FP=0, FN=1 => F1=2/3
    #
    # Class 1:
    # TP=2, FP=1, FN=0 => F1=4/5
    #
    # Macro = (2/3 + 4/5)/2
    expected = ((2 / 3) + (4 / 5)) / 2

    assert macro_f1(
        targets,
        predictions,
        num_classes=2,
    ) == pytest.approx(expected)


def test_macro_f1_zero_division_safe():
    targets = torch.tensor([0, 0])
    predictions = torch.tensor([0, 0])

    value = macro_f1(
        targets,
        predictions,
        num_classes=3,
    )

    assert math.isfinite(value)


def test_per_class_accuracy():
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])

    values = per_class_accuracy(
        targets,
        predictions,
        num_classes=2,
    )

    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)


def test_confusion_matrix():
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])

    matrix = confusion_matrix(
        targets,
        predictions,
        num_classes=2,
    )

    expected = torch.tensor([
        [1, 1],
        [0, 2],
    ])

    assert torch.equal(matrix, expected)


def test_average_accuracy():
    matrix = [
        [0.80, None, None, None, None],
        [0.70, 0.75, None, None, None],
        [0.65, 0.72, 0.78, None, None],
        [0.60, 0.68, 0.74, 0.80, None],
        [0.55, 0.65, 0.70, 0.75, 0.82],
    ]

    expected = (
        0.55
        + 0.65
        + 0.70
        + 0.75
        + 0.82
    ) / 5

    assert average_accuracy(matrix) == pytest.approx(expected)


def test_average_forgetting():
    matrix = [
        [0.80, None, None, None, None],
        [0.70, 0.75, None, None, None],
        [0.65, 0.72, 0.78, None, None],
        [0.60, 0.68, 0.74, 0.80, None],
        [0.55, 0.65, 0.70, 0.75, 0.82],
    ]

    expected = (
        (0.80 - 0.55)
        + (0.75 - 0.65)
        + (0.78 - 0.70)
        + (0.80 - 0.75)
    ) / 4

    assert average_forgetting(matrix) == pytest.approx(expected)


def test_average_forgetting_rejects_incomplete_matrix():
    matrix = [
        [0.8, None],
        [0.7, 0.75],
    ]

    with pytest.raises(ValueError):
        average_forgetting(matrix)


def test_average_accuracy_rejects_missing_final_values():
    matrix = [
        [0.8, None, None, None, None],
        [0.7, 0.75, None, None, None],
        [0.65, 0.72, 0.78, None, None],
        [0.60, 0.68, 0.74, 0.80, None],
        [0.55, None, 0.70, 0.75, 0.82],
    ]

    with pytest.raises(ValueError):
        average_accuracy(matrix)