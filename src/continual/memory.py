"""
Replay memory abstractions for RF-CL.

Person 3 ownership:
- Replay memory
- Continual-learning methods
- Evaluation
- Ablation

The memory stores training examples from previously completed tasks.

Each memory entry has:
    x    : Tensor [2, 128]
    y    : modulation class index
    snr  : original SNR value
    task : continual-learning task ID

Important:
    Validation and test samples must never be inserted into this memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class ReplaySample:
    """Single training sample stored in replay memory."""

    x: torch.Tensor
    y: int
    snr: int
    task: int

    def __post_init__(self) -> None:
        if not isinstance(self.x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")

        if self.x.shape != (2, 128):
            raise ValueError(
                f"Expected x shape [2, 128], got {tuple(self.x.shape)}."
            )

        if not isinstance(self.y, int):
            raise TypeError("y must be an int.")

        if not 0 <= self.y <= 10:
            raise ValueError(f"y must be in [0, 10], got {self.y}.")

        if not isinstance(self.snr, int):
            raise TypeError("snr must be an int.")

        if not isinstance(self.task, int):
            raise TypeError("task must be an int.")

        if not 1 <= self.task <= 5:
            raise ValueError(f"task must be in [1, 5], got {self.task}.")

    def as_dict(self) -> Dict[str, Any]:
        """Return the sample using the shared project data contract."""

        return {
            "x": self.x,
            "y": self.y,
            "snr": self.snr,
            "task": self.task,
        }


class ReplayMemory:
    """
    Fixed-capacity replay memory.

    The memory itself provides generic storage. Selection/replacement
    policies are implemented by continual-learning methods.

    Parameters
    ----------
    capacity:
        Maximum number of samples that may be stored.
    """

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int):
            raise TypeError("capacity must be an int.")

        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")

        self.capacity = capacity
        self._samples: List[ReplaySample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def len(self) -> int:
        """Explicit length accessor."""

        return len(self._samples)

    def is_empty(self) -> bool:
        return len(self._samples) == 0

    def is_full(self) -> bool:
        return len(self._samples) >= self.capacity

    def clear(self) -> None:
        """Remove all stored samples."""

        self._samples.clear()

    def add(self, sample: ReplaySample) -> None:
        """
        Add a sample.

        Raises
        ------
        OverflowError
            If adding the sample would exceed capacity.

        Notes
        -----
        Policy-specific replacement should happen before calling add().
        """

        if not isinstance(sample, ReplaySample):
            raise TypeError("sample must be a ReplaySample instance.")

        if self.is_full():
            raise OverflowError(
                "Replay memory is full. Use a selection/replacement policy "
                "before adding another sample."
            )

        # Detach storage from a potentially mutable training tensor.
        stored = ReplaySample(
            x=sample.x.detach().cpu().clone(),
            y=sample.y,
            snr=sample.snr,
            task=sample.task,
        )

        self._samples.append(stored)

    def add_training_sample(self, sample: Dict[str, Any]) -> None:
        """
        Add a sample represented using the shared data contract.

        This method is intended only for training samples.
        """

        self.add(self.from_dict(sample))

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        """Add multiple samples without exceeding capacity."""

        for sample in samples:
            self.add(sample)

    def retrieve(self) -> List[ReplaySample]:
        """
        Return a copy of the current memory contents.

        The underlying list cannot be mutated through the returned object.
        """

        return list(self._samples)

    def sample(
        self,
        indices: Sequence[int],
    ) -> List[ReplaySample]:
        """Retrieve samples by index."""

        if not indices:
            return []

        result: List[ReplaySample] = []

        for index in indices:
            if not 0 <= index < len(self._samples):
                raise IndexError(f"Memory index out of range: {index}")

            result.append(self._samples[index])

        return result

    def metadata(self) -> Dict[str, Any]:
        """Return lightweight information about memory contents."""

        return {
            "capacity": self.capacity,
            "size": len(self),
            "tasks": sorted({sample.task for sample in self._samples}),
            "snrs": sorted({sample.snr for sample in self._samples}),
            "classes": sorted({sample.y for sample in self._samples}),
        }

    def state(self) -> Dict[str, Any]:
        """Alias for metadata(), useful for experiment logging."""

        return self.metadata()

    @staticmethod
    def from_dict(sample: Dict[str, Any]) -> ReplaySample:
        """Convert the shared dataset contract to ReplaySample."""

        required = {"x", "y", "snr", "task"}

        missing = required.difference(sample.keys())

        if missing:
            raise ValueError(
                f"Sample is missing required fields: {sorted(missing)}"
            )

        return ReplaySample(
            x=sample["x"],
            y=int(sample["y"]),
            snr=int(sample["snr"]),
            task=int(sample["task"]),
        )

    @staticmethod
    def to_dict(sample: ReplaySample) -> Dict[str, Any]:
        """Convert ReplaySample back to shared dataset contract."""

        return sample.as_dict()