"""Run the two mandatory baselines: Naive Sequential and Joint Training.

Usage
-----
    python experiments/run_baselines.py
    python experiments/run_baselines.py --data synthetic --epochs 5 --verbose
    python experiments/run_baselines.py --config config.yaml --output-dir results

Both methods share:
  * the same seed,
  * the same fixed dataset splits,
  * the same backbone (a fresh model each),
  * the same training settings from config.yaml.

Nothing is hard-coded as a "result": every number is computed at run time and
written to ``results/``.

Data source
-----------
By default this uses SYNTHETIC tasks (``src/training/synthetic.py``) so the
pipeline runs before Person 1's real loader exists. When ``src/data/`` provides
a task builder, pass ``--data real`` and wire it up in ``build_real_tasks``
below -- the model API does not change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# --- Make 'src' importable regardless of the current working directory -------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn import build_model, resolve_num_classes  # noqa: E402
from src.training.seed import set_seed                       # noqa: E402
from src.training.trainer import build_loader, get_device, save_checkpoint  # noqa: E402
from src.training.synthetic import make_synthetic_tasks      # noqa: E402
from src.continual.naive import run_naive_sequential         # noqa: E402
from src.continual.joint import run_joint_training           # noqa: E402


# --------------------------------------------------------------------------- #
# Config / data
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_task_loaders(task_datasets, config: dict, seed: int) -> List[Dict]:
    """Wrap per-task datasets into the loader dicts the baselines expect.

    The SAME loader objects (built once from the same splits) are handed to both
    baselines, guaranteeing an apples-to-apples comparison.
    """
    loaders = []
    for t in task_datasets:
        loaders.append(
            {
                "train": build_loader(t["train"], config, shuffle=True, seed=seed),
                "val": build_loader(t["val"], config, shuffle=False, seed=seed)
                if t.get("val") is not None else None,
                "test": build_loader(t["test"], config, shuffle=False, seed=seed),
            }
        )
    return loaders


def build_real_tasks(config: dict):
    """Build the five real RadioML task datasets using src.data."""
    from src.data import get_task_dataset

    task_ids = sorted(
    int(str(task_id).replace("task_", ""))
    for task_id in config["tasks"].keys()
)

    return [
        {
            "train": get_task_dataset(task_id, "train"),
            "val": get_task_dataset(task_id, "val"),
            "test": get_task_dataset(task_id, "test"),
        }
        for task_id in task_ids
    ]

# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def save_accuracy_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = matrix.shape[0]
    header = "," + ",".join(f"Test_T{j + 1}" for j in range(n))
    lines = [header]
    for i in range(n):
        cells = []
        for j in range(n):
            v = matrix[i, j]
            cells.append("" if np.isnan(v) else f"{v:.6f}")
        lines.append(f"After_T{i + 1}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def print_accuracy_matrix(matrix: np.ndarray) -> None:
    n = matrix.shape[0]
    col_w = 9
    header = " " * 10 + "".join(f"Test T{j + 1}".rjust(col_w) for j in range(n))
    print(header)
    for i in range(n):
        row = f"After T{i + 1}".ljust(10)
        for j in range(n):
            v = matrix[i, j]
            row += ("" if np.isnan(v) else f"{v:.3f}").rjust(col_w)
        print(row)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run RF-CL baselines.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--num-tasks", type=int, default=5,
                        help="Number of tasks for the SYNTHETIC generator. The "
                             "real pipeline defines tasks from config['tasks'].")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs (handy for quick smoke runs).")
    parser.add_argument("--output-dir", default=None,
                        help="Override config output_dir.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    out_dir = Path(args.output_dir or config.get("output_dir", "results"))
    device = get_device()

    print(f"device={device}  seed={seed}  data={args.data}  "
          f"epochs={config.get('training', {}).get('epochs')}")

    # ---- Build fixed splits ONCE, shared across both baselines ---------------
    set_seed(seed)
    if args.data == "synthetic":
        task_datasets = make_synthetic_tasks(num_tasks=args.num_tasks,
                                             num_classes=resolve_num_classes(config),
                                             seed=seed)
    else:
        task_datasets = build_real_tasks(config)
    task_loaders = build_task_loaders(task_datasets, config, seed)

    summary: Dict[str, dict] = {}

    # ===================== Baseline 1: Naive Sequential ======================
    set_seed(seed)                       # identical init for both methods
    naive_model = build_model(config)
    print(f"\nmodel parameters: {naive_model.num_parameters():,}")
    naive_res = run_naive_sequential(naive_model, task_loaders, config,
                                     device=device, verbose=args.verbose)

    print("\n[Naive Sequential] accuracy matrix (rows: after task, cols: test set)")
    print_accuracy_matrix(naive_res["accuracy_matrix"])

    naive_dir = out_dir / "naive_sequential"
    save_checkpoint(str(naive_dir / "model.pt"), naive_model,
                    extra={"method": "naive_sequential", "seed": seed})
    save_accuracy_matrix_csv(naive_dir / "accuracy_matrix.csv",
                             naive_res["accuracy_matrix"])
    save_json(naive_dir / "metrics.json", {
        "method": "naive_sequential",
        "seed": seed,
        "num_tasks": naive_res["num_tasks"],
        "accuracy_matrix": naive_res["accuracy_matrix_list"],
        "final_accuracies": naive_res["final_accuracies"],
        "histories": naive_res["histories"],
    })
    summary["naive_sequential"] = {
        "final_accuracies": naive_res["final_accuracies"],
        "mean_final_accuracy": float(np.mean(naive_res["final_accuracies"])),
    }

    # ===================== Baseline 2: Joint Training ========================
    set_seed(seed)                       # fresh model, same init
    joint_model = build_model(config)
    joint_res = run_joint_training(joint_model, task_loaders, config,
                                   device=device, seed=seed, verbose=args.verbose)

    print("\n[Joint Training] per-task test accuracy:")
    for j, a in enumerate(joint_res["per_task_accuracy"]):
        print(f"  Test T{j + 1}: {a:.3f}")
    print(f"  mean: {joint_res['mean_accuracy']:.3f}")

    joint_dir = out_dir / "joint_training"
    save_checkpoint(str(joint_dir / "model.pt"), joint_model,
                    extra={"method": "joint_training", "seed": seed})
    save_json(joint_dir / "metrics.json", {
        "method": "joint_training",
        "seed": seed,
        "num_tasks": joint_res["num_tasks"],
        "per_task_accuracy": joint_res["per_task_accuracy"],
        "mean_accuracy": joint_res["mean_accuracy"],
        "history": joint_res["history"],
    })
    summary["joint_training"] = {
        "per_task_accuracy": joint_res["per_task_accuracy"],
        "mean_accuracy": joint_res["mean_accuracy"],
    }

    # ---- Combined summary ---------------------------------------------------
    save_json(out_dir / "summary.json", {"seed": seed, "data": args.data,
                                         "results": summary})
    print(f"\nSaved checkpoints and metrics under: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
