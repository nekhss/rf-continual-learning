"""
Domain-incremental task definitions for RF-CL.

Four tasks are defined purely in terms of *SNR ranges*. The label space
(the 11 modulation classes) is identical across all tasks -- only the
input distribution (SNR / noise level) changes. This is what makes the
benchmark domain-incremental rather than class-incremental: a class-
incremental split would instead partition the 11 *classes* across tasks,
which we deliberately do not do here.

Task 1 (highest SNR / easiest):   10, 12, 14, 16, 18 dB
Task 2:                             0,  2,  4,  6,  8 dB
Task 3:                           -10, -8, -6, -4, -2 dB
Task 4 (lowest SNR / hardest):   -20,-18,-16,-14,-12 dB
"""

from typing import Dict, List

TASK_SNR_RANGES: Dict[int, List[int]] = {
    1: [10, 12, 14, 16, 18],
    2: [0, 2, 4, 6, 8],
    3: [-10, -8, -6, -4, -2],
    4: [-20, -18, -16, -14, -12],
}

NUM_TASKS = len(TASK_SNR_RANGES)

# Full expected SNR grid for RadioML2016.10a (-20 to +18 dB, 2 dB steps).
# Used only for *validation/reporting* against what's actually found in the
# downloaded dataset -- never for hard-coding sample counts.
EXPECTED_SNR_VALUES: List[int] = sorted(
    snr for snrs in TASK_SNR_RANGES.values() for snr in snrs
)

# The 11 RadioML2016.10a modulation classes, canonical (sorted) form.
# Used only to validate what's found in the dataset -- the actual class
# list used at runtime always comes from inspecting the data itself.
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
    """Map a raw SNR value (dB) to its task id (1-4). Raises if the SNR
    does not belong to any defined task (e.g. dataset has extra SNRs we
    were not told to include)."""
    for task_id, snrs in TASK_SNR_RANGES.items():
        if snr in snrs:
            return task_id
    raise ValueError(
        f"SNR={snr} is not assigned to any of the 4 defined tasks "
        f"({TASK_SNR_RANGES}). If the dataset contains SNR values outside "
        f"this grid, decide explicitly whether to extend a task's range or "
        f"exclude those examples -- do not silently drop or reassign them."
    )


def validate_task_definition() -> None:
    """Sanity-check the task definition itself (not the dataset)."""
    assert len(TASK_SNR_RANGES) == 4, "Expected exactly 4 tasks."
    seen = set()
    for task_id, snrs in TASK_SNR_RANGES.items():
        assert len(snrs) == 5, f"Task {task_id} expected 5 SNR values, got {len(snrs)}"
        for s in snrs:
            assert s not in seen, f"SNR {s} assigned to more than one task!"
            seen.add(s)
