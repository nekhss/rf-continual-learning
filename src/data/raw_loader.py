"""
Loading and inspecting the raw RadioML2016.10a dataset.

RadioML2016.10a is distributed as a single Python-2 pickle file
(commonly named ``RML2016.10a_dict.pkl``) containing a dict:

    { (modulation_name: str, snr: int) : np.ndarray of shape (N, 2, 128) }

We never assume the exact shape/keys/counts blindly -- ``inspect_raw``
reports what is actually present, and ``load_master_arrays`` builds a
single deterministic, concatenated array representation from whatever
was found, adapting to the real structure rather than forcing assumptions.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Candidate locations we'll look for the dataset in, in order.
DEFAULT_CANDIDATE_PATHS = [
    "data/raw/RML2016.10a_dict.pkl",
    "data/raw/RML2016.10a_dict.pkl.tar.bz2",
    "/mnt/user-data/uploads/RML2016.10a_dict.pkl",
]


class DatasetNotFoundError(FileNotFoundError):
    pass


def find_raw_dataset(candidate_paths: List[str] = None) -> str:
    """Locate the raw RadioML2016.10a pickle file on disk.

    We deliberately do NOT auto-download from the network here: this
    dataset is normally hosted on deepsig.ai / opendata mirrors, and this
    project's sandboxed environments frequently cannot reach those hosts.
    Failing loudly with clear instructions is safer than silently
    substituting something else.
    """
    candidate_paths = candidate_paths or DEFAULT_CANDIDATE_PATHS
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    raise DatasetNotFoundError(
        "Could not find RadioML2016.10a. Expected the pickle file "
        f"(commonly 'RML2016.10a_dict.pkl') at one of: {candidate_paths}. "
        "Download it manually (e.g. from the DeepSig RadioML2016.10a "
        "release) and place it at 'data/raw/RML2016.10a_dict.pkl', then "
        "re-run this script."
    )


def load_raw_dict(path: str) -> Dict[Tuple[str, int], np.ndarray]:
    """Load the raw {(mod, snr): array} dict from a RadioML2016.10a pickle.

    The reference file was pickled under Python 2, so we need
    encoding='latin1' to unpickle it under Python 3.
    """
    with open(path, "rb") as f:
        try:
            raw = pickle.load(f, encoding="latin1")
        except TypeError:
            # Already-Python-3-pickled files (e.g. our synthetic fixtures)
            # don't accept the encoding kwarg on some pickle protocols.
            f.seek(0)
            raw = pickle.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected the RadioML pickle to contain a dict, got {type(raw)}. "
            "The dataset format differs from what this pipeline assumes -- "
            "inspect the file manually before proceeding."
        )
    return raw


@dataclass
class DatasetReport:
    num_examples: int
    class_names: List[str]
    class_counts: Dict[str, int]
    snr_values: List[int]
    snr_counts: Dict[int, int]
    sample_shape: Tuple[int, ...]
    key_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Total examples: {self.num_examples}",
            f"Classes ({len(self.class_names)}): {self.class_names}",
            f"SNR values ({len(self.snr_values)}): {self.snr_values}",
            f"Per-sample shape: {self.sample_shape}",
            "Class balance (examples per class, across all SNRs):",
        ]
        for c in self.class_names:
            lines.append(f"  {c:10s}: {self.class_counts[c]}")
        lines.append("SNR distribution (examples per SNR, across all classes):")
        for s in self.snr_values:
            lines.append(f"  {s:+4d} dB : {self.snr_counts[s]}")
        return lines


def inspect_raw(raw: Dict[Tuple[str, int], np.ndarray]) -> DatasetReport:
    """Inspect the raw dict and report what's actually in it. Makes no
    assumption about class names, SNR values, or example counts -- those
    are all *discovered* here, then validated separately against what the
    task expects."""
    if len(raw) == 0:
        raise ValueError("Raw dataset dict is empty.")

    keys = list(raw.keys())
    class_names = sorted({k[0] for k in keys})
    snr_values = sorted({int(k[1]) for k in keys})

    sample_shapes = {tuple(arr.shape[1:]) for arr in raw.values()}
    if len(sample_shapes) != 1:
        raise ValueError(
            f"Inconsistent per-sample shapes found across (mod, snr) groups: "
            f"{sample_shapes}. Dataset does not match the assumed uniform "
            f"[channels, samples] structure -- inspect it manually."
        )
    sample_shape = next(iter(sample_shapes))

    key_counts = {k: int(v.shape[0]) for k, v in raw.items()}
    num_examples = sum(key_counts.values())

    class_counts = {c: 0 for c in class_names}
    snr_counts = {s: 0 for s in snr_values}
    for (mod, snr), count in key_counts.items():
        class_counts[mod] += count
        snr_counts[int(snr)] += count

    return DatasetReport(
        num_examples=num_examples,
        class_names=class_names,
        class_counts=class_counts,
        snr_values=snr_values,
        snr_counts=snr_counts,
        sample_shape=sample_shape,
        key_counts=key_counts,
    )


def validate_report(report: DatasetReport, expected_class_names, expected_snr_values) -> List[str]:
    """Compare discovered structure against expectations. Returns a list
    of human-readable discrepancy messages (empty list = everything matches).
    Never raises and never silently forces data to match -- callers decide
    what to do with discrepancies."""
    issues = []

    found_classes = set(report.class_names)
    expected_classes = set(expected_class_names)
    if found_classes != expected_classes:
        missing = expected_classes - found_classes
        extra = found_classes - expected_classes
        if missing:
            issues.append(f"Missing expected classes: {sorted(missing)}")
        if extra:
            issues.append(f"Unexpected extra classes found: {sorted(extra)}")

    found_snrs = set(report.snr_values)
    expected_snrs = set(expected_snr_values)
    if found_snrs != expected_snrs:
        missing = expected_snrs - found_snrs
        extra = found_snrs - expected_snrs
        if missing:
            issues.append(f"Missing expected SNR values: {sorted(missing)}")
        if extra:
            issues.append(
                f"Unexpected extra SNR values found (not covered by any "
                f"task): {sorted(extra)}"
            )

    if report.sample_shape != (2, 128):
        issues.append(
            f"Expected per-sample shape (2, 128) [I/Q x 128 samples], "
            f"found {report.sample_shape}."
        )

    return issues


def load_master_arrays(
    raw: Dict[Tuple[str, int], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Flatten the raw {(mod, snr): array} dict into master arrays using a
    fully deterministic ordering (sorted by class name, then by SNR, then
    by original in-array order). This ordering -- not any incidental Python
    dict ordering -- is what split indices in splits_seed42.json refer to,
    so re-running this function against the same raw file always
    reproduces the exact same global index for every example.

    Returns:
        X:   float32 array, shape [N, 2, 128]
        y:   int64 array, shape [N], values in [0, num_classes)
        snr: int64 array, shape [N], raw SNR in dB
        class_names: sorted list of class names; class_names[y[i]] is the
                     modulation of example i.
    """
    class_names = sorted({k[0] for k in raw.keys()})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    snr_values = sorted({int(k[1]) for k in raw.keys()})

    X_parts, y_parts, snr_parts = [], [], []
    for c in class_names:
        for s in snr_values:
            key = (c, s)
            if key not in raw:
                continue  # dataset need not be a dense grid; adapt, don't force
            arr = np.asarray(raw[key], dtype=np.float32)
            n = arr.shape[0]
            X_parts.append(arr)
            y_parts.append(np.full(n, class_to_idx[c], dtype=np.int64))
            snr_parts.append(np.full(n, s, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    snr = np.concatenate(snr_parts, axis=0)
    return X, y, snr, class_names
