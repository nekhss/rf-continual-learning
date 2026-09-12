"""
Domain-incremental task definitions for RF-CL.

The benchmark contains five sequential SNR domains. All tasks share the
same 11-class modulation label space; only the SNR/input distribution
changes.

Task 1: [12, 14, 16, 18] dB
Task 2: [4, 6, 8, 10] dB
Task 3: [-4, -2, 0, 2] dB
Task 4: [-12, -10, -8, -6] dB
Task 5: [-20, -18, -16, -14] dB

This is domain-incremental learning, not class-incremental learning.
"""

from typing import Dict, List


TASK_SNR_RANGES: Dict[int, List[int]] = {
    1: [12, 14, 16, 18],
    2: [4, 6, 8, 10],
    3: [-4, -2, 0, 2],
    4: [-12, -10, -8, -6],
    5: [-20, -18, -16, -14],
}

NUM_TASKS = len(TASK_SNR_RANGES)

EXPECTED_SNR_VALUES: List[int] = sorted(
    snr for snrs in TASK_SNR_RANGES.values() for snr in snrs
)

EXPECTED_CLASS_NAMES: List[str] = sorted(
    [
        "8PSK",
        "AM-DSB",
        "AM-SSB",
        "BPSK",
        "CPFSK",
        "GFSK",
        "PAM4",
        "QAM16",
        "QAM64",
        "QPSK",
        "WBFM",
    ]
)


def snr_to_task(snr: int) -> int:
    """Map an SNR value in dB to its RF-CL task ID."""
    for task_id, snrs in TASK_SNR_RANGES.items():
        if snr in snrs:
            return task_id

    raise ValueError(
        f"SNR={snr} is not assigned to any RF-CL task "
        f"({TASK_SNR_RANGES})."
    )


def validate_task_definition() -> None:
    """Validate the five-task RF-CL benchmark definition."""
    assert NUM_TASKS == 5, f"Expected 5 tasks, got {NUM_TASKS}"

    seen = set()

    for task_id, snrs in TASK_SNR_RANGES.items():
        assert len(snrs) == 4, (
            f"Task {task_id} expected 4 SNR values, got {len(snrs)}"
        )

        for snr in snrs:
            assert snr not in seen, (
                f"SNR {snr} assigned to more than one task"
            )
            seen.add(snr)

    assert seen == set(range(-20, 20, 2)), (
        f"Unexpected SNR coverage: {sorted(seen)}"
    )