"""
RadioML2016.10a data preparation for RF-CL.

Run with:
    python src/data/prepare_data.py

What it does:
    1. Locates the raw RadioML2016.10a pickle on disk (does not download --
       see README_DATA.md for why, and where to put the file).
    2. Inspects the actual dataset contents (classes, SNRs, counts, shape).
    3. Validates that against what the 4-task design expects, reporting
       (not silently hiding) any discrepancy.
    4. Builds the 4 domain-incremental tasks.
    5. Computes deterministic, class-stratified train/val/test splits
       (seed 42) independently per task.
    6. Runs the data-integrity checks (no leakage across splits/tasks).
    7. Saves the split indices to data/splits/splits_seed42.json.
    8. Prints a human-readable summary.
"""

import os
import sys

# Allow `python src/data/prepare_data.py` to work directly (as required)
# by ensuring the repo root -- not this file's directory -- is on sys.path,
# so `import src.data...` resolves regardless of invocation style.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.raw_loader import (
    find_raw_dataset,
    load_raw_dict,
    inspect_raw,
    validate_report,
    load_master_arrays,
)
from src.data.tasks import (
    TASK_SNR_RANGES,
    EXPECTED_CLASS_NAMES,
    EXPECTED_SNR_VALUES,
    validate_task_definition,
)
from src.data.splits import build_task_splits, save_splits, validate_splits, SEED

SPLITS_OUTPUT_PATH = "data/splits/splits_seed42.json"


def main() -> int:
    print("=" * 40)
    print("RadioML 2016.10a DATA PREPARATION")
    print("=" * 40)
    print()
    print(f"Seed: {SEED}")

    validate_task_definition()

    # 1. Locate + load raw dataset.
    try:
        raw_path = find_raw_dataset()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1
    print(f"Raw dataset: {raw_path}")

    raw = load_raw_dict(raw_path)

    # 2. Inspect.
    report = inspect_raw(raw)
    print(f"Classes: {len(report.class_names)}")
    print(f"Samples: {report.num_examples}")
    print()

    # 3. Validate against expectations -- report, don't silently force.
    issues = validate_report(report, EXPECTED_CLASS_NAMES, EXPECTED_SNR_VALUES)
    if issues:
        print("!" * 40)
        print("DATASET DISCREPANCIES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print(
            "Proceeding using the dataset's ACTUAL discovered classes/SNRs "
            "below rather than forcing the originally-assumed values. "
            "Verify these discrepancies are expected before trusting results."
        )
        print("!" * 40)
        print()

    # 4. Build master arrays (deterministic global ordering) + tasks.
    X, y, snr, class_names = load_master_arrays(raw)

    for task_id, snr_list in TASK_SNR_RANGES.items():
        lo, hi = min(snr_list), max(snr_list)
        task_mask = [s in snr_list for s in snr]
        task_total = sum(task_mask)
        print(f"Task {task_id}: {lo} to {hi} dB")
        print(f"  Total: {task_total}")

    print()

    # 5. Deterministic stratified splits, per task.
    splits = build_task_splits(y, snr, seed=SEED)

    for task_id, snr_list in TASK_SNR_RANGES.items():
        lo, hi = min(snr_list), max(snr_list)
        s = splits[task_id]
        print(f"Task {task_id}: {lo} to {hi} dB")
        print(f"  Train: {len(s['train'])}")
        print(f"  Val:   {len(s['val'])}")
        print(f"  Test:  {len(s['test'])}")

    print()

    # 6. Integrity checks -- fail loudly rather than saving bad splits.
    problems = validate_splits(splits)
    if problems:
        print("FATAL: split integrity checks failed:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Integrity checks passed: no train/val/test overlap, "
          "no example shared across tasks.")
    print()

    print("Class balance (examples per class, whole dataset):")
    for c in class_names:
        print(f"  {c:10s}: {report.class_counts.get(c, 'n/a')}")
    print()
    print("SNR distribution (examples per SNR, whole dataset):")
    for s_val in report.snr_values:
        print(f"  {s_val:+4d} dB : {report.snr_counts[s_val]}")
    print()

    # 7. Save.
    save_splits(splits, class_names, SPLITS_OUTPUT_PATH, seed=SEED)
    print(f"Splits saved to:\n{SPLITS_OUTPUT_PATH}")
    print()
    print("=" * 40)
    return 0


if __name__ == "__main__":
    sys.exit(main())
