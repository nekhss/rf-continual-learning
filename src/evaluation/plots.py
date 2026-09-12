"""
Plotting utilities for RF-CL.

Required figures:
    1. Random Replay accuracy matrix
    2. SNR-Aware Replay accuracy matrix
    3. Task-1 accuracy across training stages
    4. Final Average Accuracy comparison
    5. Average Forgetting comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _prepare_matrix(
    matrix: Sequence[Sequence[float | None]],
) -> np.ndarray:
    """Convert a triangular accuracy matrix to NaN-masked NumPy array."""

    result = np.full(
        (len(matrix), len(matrix)),
        np.nan,
        dtype=float,
    )

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value is not None:
                result[i, j] = float(value)

    return result


def plot_accuracy_matrix(
    matrix: Sequence[Sequence[float | None]],
    method: str,
    output_path: str | Path,
) -> None:
    """Plot one continual-learning accuracy matrix."""

    data = _prepare_matrix(matrix)

    fig, ax = plt.subplots(figsize=(7, 6))

    masked = np.ma.masked_invalid(data)

    image = ax.imshow(
        masked,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_title(f"{method} Accuracy Matrix")
    ax.set_xlabel("Test Task")
    ax.set_ylabel("Model Trained Through Task")

    num_tasks = len(matrix)

    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels(
        [f"T{i}" for i in range(1, num_tasks + 1)]
    )

    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels(
        [f"After T{i}" for i in range(1, num_tasks + 1)]
    )

    for i in range(num_tasks):
        for j in range(num_tasks):
            if not np.isnan(data[i, j]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.3f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        image,
        ax=ax,
        label="Accuracy",
    )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_task1_accuracy(
    random_matrix: Sequence[Sequence[float | None]],
    snr_aware_matrix: Sequence[Sequence[float | None]],
    output_path: str | Path,
) -> None:
    """
    Plot Task-1 accuracy as training progresses.

    Only valid observations A11, A21, ..., A51 are plotted.
    """

    random_values = [
        row[0]
        for row in random_matrix
        if len(row) > 0 and row[0] is not None
    ]

    snr_values = [
        row[0]
        for row in snr_aware_matrix
        if len(row) > 0 and row[0] is not None
    ]

    stages = list(range(1, max(
        len(random_values),
        len(snr_values),
    ) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))

    if random_values:
        ax.plot(
            stages[:len(random_values)],
            random_values,
            marker="o",
            label="Random Replay",
        )

    if snr_values:
        ax.plot(
            stages[:len(snr_values)],
            snr_values,
            marker="o",
            label="SNR-Aware Replay",
        )

    ax.set_title("Task-1 Accuracy Across Training Stages")
    ax.set_xlabel("Training Through Task")
    ax.set_ylabel("Task-1 Test Accuracy")
    ax.set_xticks(stages)
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_average_accuracy(
    results: Dict[str, float],
    output_path: str | Path,
) -> None:
    """Plot final Average Accuracy comparison."""

    methods = list(results.keys())
    values = [results[method] for method in methods]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(methods, values)

    ax.set_title("Final Average Accuracy")
    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0.0, 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_average_forgetting(
    results: Dict[str, float],
    output_path: str | Path,
) -> None:
    """Plot Average Forgetting comparison."""

    methods = list(results.keys())
    values = [results[method] for method in methods]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(methods, values)

    ax.set_title("Average Forgetting")
    ax.set_ylabel("Average Forgetting")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)