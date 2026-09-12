import torch
import pytest

from src.continual.memory import ReplayMemory, ReplaySample
from src.continual.replay import RandomReplay, SNRAwareReplay


def make_sample(
    task: int,
    snr: int,
    y: int,
) -> ReplaySample:
    return ReplaySample(
        x=torch.randn(2, 128),
        y=y,
        snr=snr,
        task=task,
    )


def make_task_samples(task: int, n: int = 30):
    snrs = {
        1: [12, 14, 16, 18],
        2: [4, 6, 8, 10],
        3: [-4, -2, 0, 2],
        4: [-12, -10, -8, -6],
        5: [-20, -18, -16, -14],
    }

    return [
        make_sample(
            task=task,
            snr=snrs[task][i % 4],
            y=i % 11,
        )
        for i in range(n)
    ]


def test_memory_starts_empty():
    memory = ReplayMemory(capacity=10)

    assert len(memory) == 0
    assert memory.is_empty()


def test_memory_accepts_valid_samples():
    memory = ReplayMemory(capacity=10)

    sample = make_sample(1, 12, 0)

    memory.add(sample)

    assert len(memory) == 1

    stored = memory.retrieve()[0]

    assert stored.y == 0
    assert stored.snr == 12
    assert stored.task == 1
    assert stored.x.shape == (2, 128)


def test_memory_never_exceeds_capacity():
    memory = ReplayMemory(capacity=3)

    for i in range(3):
        memory.add(make_sample(1, 12, i))

    with pytest.raises(OverflowError):
        memory.add(make_sample(1, 12, 3))

    assert len(memory) == 3


def test_memory_stores_task_and_snr_metadata():
    memory = ReplayMemory(capacity=10)

    memory.add(make_sample(2, 6, 4))

    metadata = memory.metadata()

    assert metadata["tasks"] == [2]
    assert metadata["snrs"] == [6]
    assert metadata["classes"] == [4]


def test_invalid_sample_shape_rejected():
    with pytest.raises(ValueError):
        ReplaySample(
            x=torch.randn(3, 128),
            y=0,
            snr=12,
            task=1,
        )


def test_invalid_class_rejected():
    with pytest.raises(ValueError):
        ReplaySample(
            x=torch.randn(2, 128),
            y=11,
            snr=12,
            task=1,
        )


def test_random_replay_only_contains_seen_tasks():
    replay = RandomReplay(
        memory_size=20,
        seed=42,
    )

    replay.update(make_task_samples(1))

    assert {
        sample.task
        for sample in replay.get_memory().retrieve()
    } == {1}

    replay.update(make_task_samples(2))

    assert {
        sample.task
        for sample in replay.get_memory().retrieve()
    } <= {1, 2}


def test_random_replay_never_contains_future_tasks():
    replay = RandomReplay(
        memory_size=20,
        seed=42,
    )

    replay.update(make_task_samples(1))
    replay.update(make_task_samples(2))
    replay.update(make_task_samples(3))

    tasks = {
        sample.task
        for sample in replay.get_memory().retrieve()
    }

    assert tasks <= {1, 2, 3}
    assert 4 not in tasks
    assert 5 not in tasks


def test_random_replay_deterministic():
    task1 = make_task_samples(1)
    task2 = make_task_samples(2)

    replay_a = RandomReplay(10, seed=42)
    replay_b = RandomReplay(10, seed=42)

    replay_a.update(task1)
    replay_a.update(task2)

    replay_b.update(task1)
    replay_b.update(task2)

    a = [
        (sample.y, sample.snr, sample.task)
        for sample in replay_a.get_memory().retrieve()
    ]

    b = [
        (sample.y, sample.snr, sample.task)
        for sample in replay_b.get_memory().retrieve()
    ]

    assert a == b


def test_snr_aware_capacity():
    replay = SNRAwareReplay(
        memory_size=20,
        seed=42,
    )

    for task in range(1, 6):
        replay.update(make_task_samples(task, 40))

        assert len(replay.get_memory()) <= 20


def test_snr_aware_preserves_metadata():
    replay = SNRAwareReplay(
        memory_size=20,
        seed=42,
    )

    replay.update(make_task_samples(1, 50))

    for sample in replay.get_memory().retrieve():
        assert 0 <= sample.y <= 10
        assert sample.snr in [12, 14, 16, 18]
        assert sample.task == 1


def test_snr_aware_deterministic():
    task1 = make_task_samples(1, 50)
    task2 = make_task_samples(2, 50)

    replay_a = SNRAwareReplay(20, seed=42)
    replay_b = SNRAwareReplay(20, seed=42)

    replay_a.update(task1)
    replay_a.update(task2)

    replay_b.update(task1)
    replay_b.update(task2)

    a = [
        (sample.y, sample.snr, sample.task)
        for sample in replay_a.get_memory().retrieve()
    ]

    b = [
        (sample.y, sample.snr, sample.task)
        for sample in replay_b.get_memory().retrieve()
    ]

    assert a == b


def test_snr_aware_contains_only_seen_tasks():
    replay = SNRAwareReplay(
        memory_size=20,
        seed=42,
    )

    replay.update(make_task_samples(1))
    replay.update(make_task_samples(2))
    replay.update(make_task_samples(3))

    tasks = {
        sample.task
        for sample in replay.get_memory().retrieve()
    }

    assert tasks <= {1, 2, 3}
    assert 4 not in tasks
    assert 5 not in tasks