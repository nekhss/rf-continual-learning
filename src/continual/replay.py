"""
Replay strategies for RF-CL.

Implemented methods
-------------------
1. Random Experience Replay
2. SNR-Aware Experience Replay

Both methods operate on the same ReplayMemory abstraction.

The project uses:
    seed = 42

Task ordering:
    T1 = [12, 14, 16, 18]
    T2 = [4, 6, 8, 10]
    T3 = [-4, -2, 0, 2]
    T4 = [-12, -10, -8, -6]
    T5 = [-20, -18, -16, -14]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .memory import ReplayMemory, ReplaySample


SEED = 42

TASK_SNR_MAP = {
    1: [12, 14, 16, 18],
    2: [4, 6, 8, 10],
    3: [-4, -2, 0, 2],
    4: [-12, -10, -8, -6],
    5: [-20, -18, -16, -14],
}


def set_replay_seed(seed: int = SEED) -> None:
    """Set deterministic PyTorch randomness for replay selection."""

    torch.manual_seed(seed)


def _validate_training_samples(
    samples: Iterable[ReplaySample],
) -> List[ReplaySample]:
    """Materialize and validate replay candidates."""

    result = list(samples)

    for sample in result:
        if not isinstance(sample, ReplaySample):
            raise TypeError(
                "Replay candidates must contain ReplaySample objects."
            )

    return result


def _random_indices(
    n: int,
    k: int,
    generator: torch.Generator,
) -> List[int]:
    """Deterministically select k unique indices."""

    if k <= 0 or n == 0:
        return []

    k = min(k, n)

    permutation = torch.randperm(n, generator=generator)

    return permutation[:k].tolist()


class RandomReplay:
    """
    Standard Random Experience Replay.

    At each task boundary:
        memory = randomly selected historical training samples

    The memory is bounded by `memory_size`.
    """

    def __init__(
        self,
        memory_size: int,
        seed: int = SEED,
    ) -> None:
        self.memory_size = memory_size
        self.seed = seed
        self.memory = ReplayMemory(memory_size)

    def update(
        self,
        current_task_samples: Iterable[ReplaySample],
    ) -> ReplayMemory:
        """
        Update replay memory using current task training samples.

        Only training samples from completed tasks should be supplied.
        """

        current = _validate_training_samples(current_task_samples)

        existing = self.memory.retrieve()

        candidates = existing + current

        if len(candidates) <= self.memory_size:
            selected = candidates
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

            indices = _random_indices(
                len(candidates),
                self.memory_size,
                generator,
            )

            selected = [candidates[i] for i in indices]

        self.memory.clear()

        for sample in selected:
            self.memory.add(sample)

        return self.memory

    def get_memory(self) -> ReplayMemory:
        return self.memory


class SNRAwareReplay:
    """
    SNR-aware experience replay.

    Selection explicitly considers:
        task
        SNR
        modulation class

    The policy works by forming task/SNR/class groups and performing
    deterministic round-robin selection across groups.

    This is intentionally simple and transparent enough for the report.

    It does NOT claim to be a fundamentally new continual-learning
    algorithm. The contribution is the application of SNR-aware
    experience replay to RF domain-incremental modulation classification.
    """

    def __init__(
        self,
        memory_size: int,
        seed: int = SEED,
    ) -> None:
        if memory_size <= 0:
            raise ValueError("memory_size must be positive.")

        self.memory_size = memory_size
        self.seed = seed
        self.memory = ReplayMemory(memory_size)

    def _group_samples(
        self,
        samples: Sequence[ReplaySample],
    ) -> Dict[Tuple[int, int, int], List[ReplaySample]]:
        """
        Group samples by:

            (task, snr, class)
        """

        groups: Dict[
            Tuple[int, int, int],
            List[ReplaySample],
        ] = defaultdict(list)

        for sample in samples:
            key = (
                sample.task,
                sample.snr,
                sample.y,
            )

            groups[key].append(sample)

        return dict(groups)

    def _shuffle_groups(
        self,
        groups: Dict[Tuple[int, int, int], List[ReplaySample]],
    ) -> None:
        """Deterministically shuffle samples within each group."""

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        for key in sorted(groups):
            samples = groups[key]

            if len(samples) <= 1:
                continue

            permutation = torch.randperm(
                len(samples),
                generator=generator,
            ).tolist()

            groups[key] = [samples[i] for i in permutation]

    def _select_balanced(
        self,
        candidates: Sequence[ReplaySample],
    ) -> List[ReplaySample]:
        """
        Select a bounded set while covering historical
        task/SNR/class groups as evenly as possible.
        """

        if len(candidates) <= self.memory_size:
            return list(candidates)

        groups = self._group_samples(candidates)

        self._shuffle_groups(groups)

        group_keys = sorted(groups)

        selected: List[ReplaySample] = []

        # Round-robin through task/SNR/class groups.
        #
        # This avoids filling the buffer disproportionately from
        # whichever group happens to contain the most examples.
        pointers = {key: 0 for key in group_keys}

        while len(selected) < self.memory_size:
            added_this_round = False

            for key in group_keys:
                if len(selected) >= self.memory_size:
                    break

                pointer = pointers[key]

                if pointer >= len(groups[key]):
                    continue

                selected.append(groups[key][pointer])
                pointers[key] += 1
                added_this_round = True

            if not added_this_round:
                break

        return selected

    def update(
        self,
        current_task_samples: Iterable[ReplaySample],
    ) -> ReplayMemory:
        """
        Update memory using only current-task training samples plus
        previously stored replay samples.
        """

        current = _validate_training_samples(current_task_samples)

        existing = self.memory.retrieve()

        candidates = existing + current

        selected = self._select_balanced(candidates)

        self.memory.clear()

        for sample in selected:
            self.memory.add(sample)

        return self.memory

    def get_memory(self) -> ReplayMemory:
        return self.memory


def build_training_batch(
    current_samples: Sequence[Dict],
    memory: ReplayMemory,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine current-task training samples with replay samples.

    Returns
    -------
    x:
        Tensor [N, 2, 128]

    y:
        Tensor [N]

    This helper intentionally ignores SNR/task for model input because
    the model API is simply:

        logits = model(x)
    """

    replay_samples = memory.retrieve()

    all_samples = list(current_samples) + [
        ReplayMemory.to_dict(sample)
        for sample in replay_samples
    ]

    if not all_samples:
        raise ValueError("Cannot build a training batch from zero samples.")

    xs = torch.stack([sample["x"] for sample in all_samples])

    ys = torch.tensor(
        [int(sample["y"]) for sample in all_samples],
        dtype=torch.long,
    )

    return xs, ys