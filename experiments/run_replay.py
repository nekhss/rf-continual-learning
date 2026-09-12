from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.continual.memory import ReplayMemory, ReplaySample
from src.continual.replay import RandomReplay, SNRAwareReplay
from src.evaluation.evaluate import (
    build_summary,
    evaluate_seen_tasks,
    new_accuracy_matrix,
    save_results,
    update_accuracy_matrix,
)
from src.models import build_model
from src.training.seed import set_seed
from src.training.trainer import Trainer


# ============================================================================
# Dataset helpers
# ============================================================================


class ReplayTrainingDataset(Dataset):
    """
    Dataset containing current-task samples plus replay samples.

    Samples follow the project's standard data contract:

        {
            "x": tensor [2, 128],
            "y": int,
            "snr": int,
            "task": int
        }
    """

    def __init__(self, samples: Sequence[Dict]):
        self.samples = list(samples)

        if not self.samples:
            raise ValueError("ReplayTrainingDataset cannot be empty.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]


def dataset_to_replay_samples(
    dataset: Dataset,
) -> List[ReplaySample]:
    """
    Convert a task training dataset into ReplaySample objects.

    IMPORTANT:
        Only training data should be passed here.
        Validation and test samples must never enter replay memory.
    """

    samples: List[ReplaySample] = []

    for index in range(len(dataset)):
        sample = dataset[index]

        samples.append(
            ReplaySample(
                x=sample["x"],
                y=int(sample["y"]),
                snr=int(sample["snr"]),
                task=int(sample["task"]),
            )
        )

    return samples


def replay_memory_to_dicts(
    memory: ReplayMemory,
) -> List[Dict]:
    """
    Convert replay-memory samples back into the standard
    dataset contract.
    """

    return [
        ReplayMemory.to_dict(sample)
        for sample in memory.retrieve()
    ]


def make_loader(
    dataset: Dataset,
    config: dict,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Create a deterministic DataLoader compatible with Trainer.
    """

    training_config = config["training"]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training_config.get("num_workers", 0)),
        generator=generator if shuffle else None,
    )


# ============================================================================
# Configuration / task loading
# ============================================================================


def build_real_tasks(
    config: dict,
) -> List[Dict[str, Dataset]]:
    """
    Load the five fixed RadioML 2016.10a task datasets.

    Tasks are expected to be:

        T1 = [12, 14, 16, 18]
        T2 = [4, 6, 8, 10]
        T3 = [-4, -2, 0, 2]
        T4 = [-12, -10, -8, -6]
        T5 = [-20, -18, -16, -14]
    """

    from src.data import get_task_dataset

    task_ids = sorted(
        int(str(task_id).replace("task_", ""))
        for task_id in config["tasks"].keys()
    )

    if task_ids != [1, 2, 3, 4, 5]:
        raise ValueError(
            "RF-CL replay requires exactly five tasks "
            f"[1, 2, 3, 4, 5], got {task_ids}."
        )

    return [
        {
            "train": get_task_dataset(task_id, "train"),
            "val": get_task_dataset(task_id, "val"),
            "test": get_task_dataset(task_id, "test"),
        }
        for task_id in task_ids
    ]


# ============================================================================
# Replay experiment
# ============================================================================


def run_replay_method(
    method_name: str,
    strategy,
    tasks: List[Dict[str, Dataset]],
    config: dict,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
    verbose: bool,
) -> Dict:
    """
    Run one replay method through all five sequential tasks.

    Protocol
    --------
    For each task:

        1. Load current task training data.
        2. Retrieve PREVIOUS replay memory.
        3. Train on:
               current task training data + previous replay memory
        4. Evaluate on all test tasks seen so far.
        5. Update replay memory using:
               previous memory + current task training data

    This ordering is intentional.

    The newly selected samples from the current task are NOT used
    to train that same task. They become available only for the
    following task.

    Test data is never used for training, replay, tuning, or memory.
    """

    print(f"\n{'=' * 72}")
    print(f"[{method_name}] replay experiment")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------
    # Reproducible model initialization
    # ------------------------------------------------------------------

    set_seed(seed)

    model = build_model(config)
    trainer = Trainer(
        config=config,
        device=device,
    )

    print(
        f"model parameters: "
        f"{model.num_parameters():,}"
    )

    num_tasks = len(tasks)
    num_classes = int(
        config["evaluation"]["num_classes"]
    )

    accuracy_matrix = new_accuracy_matrix(
        num_tasks=num_tasks
    )

    # Training history for each task.
    task_histories: Dict[str, Dict] = {}

    # Complete evaluation information after each task.
    #
    # Example:
    #
    # evaluation["3"]["1"]["macro_f1"]
    # evaluation["3"]["2"]["confusion_matrix"]
    # evaluation["3"]["3"]["per_class_accuracy"]
    #
    # The first key means "evaluated after training through T3".
    evaluation_history: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Sequential task loop
    # ------------------------------------------------------------------

    for task_index, task in enumerate(
        tasks,
        start=1,
    ):
        train_dataset = task["train"]
        val_dataset = task["val"]

        print(
            f"\n[{method_name}] "
            f"training task {task_index}/{num_tasks} "
            f"(current data + previous replay memory)"
        )

        # ==============================================================
        # 1. Current task training samples
        # ==============================================================

        current_samples = [
            train_dataset[index]
            for index in range(len(train_dataset))
        ]

        # ==============================================================
        # 2. Retrieve ONLY previous replay memory
        # ==============================================================

        previous_memory = strategy.get_memory()

        previous_memory_dicts = replay_memory_to_dicts(
            previous_memory
        )

        # ==============================================================
        # 3. Construct training data
        # ==============================================================

        # IMPORTANT:
        #
        # At T1:
        #     current T1 data only
        #
        # At T2:
        #     current T2 data + memory from T1
        #
        # At T3:
        #     current T3 data + memory from T1/T2
        #
        # etc.
        #
        # Current-task samples selected for replay are NOT available
        # until after this training step.

        combined_samples = (
            previous_memory_dicts
            + current_samples
        )

        train_combined = ReplayTrainingDataset(
            combined_samples
        )

        train_loader = make_loader(
            train_combined,
            config,
            shuffle=True,
            seed=seed + task_index,
        )

        # Validation remains current-task validation only.
        #
        # Replay memory must not enter validation.
        val_loader = make_loader(
            val_dataset,
            config,
            shuffle=False,
            seed=seed,
        )

        print(
            f"[{method_name} T{task_index}] "
            f"current={len(current_samples):,} "
            f"replay={len(previous_memory_dicts):,} "
            f"train_total={len(combined_samples):,}"
        )

        # ==============================================================
        # 4. Train using the project's existing Trainer
        # ==============================================================

        history = trainer.fit(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            verbose=verbose,
            log_prefix=(
                f"[{method_name} T{task_index}] "
            ),
        )

        task_histories[str(task_index)] = history

        # ==============================================================
        # 5. Evaluate ONLY on test tasks seen so far
        # ==============================================================

        seen_test_loaders: Dict[int, DataLoader] = {}

        for seen_task_index in range(task_index):
            seen_task_id = seen_task_index + 1

            test_loader = make_loader(
                tasks[seen_task_index]["test"],
                config,
                shuffle=False,
                seed=seed,
            )

            seen_test_loaders[seen_task_id] = test_loader

        seen_results = evaluate_seen_tasks(
            model=model,
            test_loaders=seen_test_loaders,
            seen_tasks=range(1, task_index + 1),
            device=device,
            num_classes=num_classes,
        )

        # --------------------------------------------------------------
        # Store COMPLETE evaluation results.
        #
        # evaluate_seen_tasks() already computes:
        #   - accuracy
        #   - macro F1
        #   - per-class accuracy
        #   - confusion matrix
        #   - number of samples
        #
        # Do not throw these away.
        # --------------------------------------------------------------

        evaluation_history[str(task_index)] = {
            str(seen_task_id): result
            for seen_task_id, result in seen_results.items()
        }

        # --------------------------------------------------------------
        # Update the continual-learning accuracy matrix.
        #
        # Only accuracy goes into the matrix because AA/AF are
        # defined from this matrix.
        # --------------------------------------------------------------

        update_accuracy_matrix(
    accuracy_matrix=accuracy_matrix,
    trained_through_task=task_index,
    task_results=seen_results,
    num_tasks=num_tasks,
)

        # --------------------------------------------------------------
        # Human-readable evaluation output
        # --------------------------------------------------------------

        print(
            f"\n[{method_name}] "
            f"after T{task_index}: "
            "test metrics on seen tasks"
        )

        for seen_task_id, result in seen_results.items():
            print(
                f"  Test T{seen_task_id}: "
                f"accuracy={result['accuracy']:.4f} "
                f"macro_f1={result['macro_f1']:.4f}"
            )

        # ==============================================================
        # 6. Update replay memory AFTER training and evaluation
        # ==============================================================

        current_replay_samples = (
            dataset_to_replay_samples(train_dataset)
        )

        # Strategy receives:
        #
        #   previous memory + current task training samples
        #
        # It decides which samples remain in the fixed-size memory.
        #
        # Validation and test samples never enter this operation.
        strategy.update(current_replay_samples)

        updated_memory = strategy.get_memory()

        print(
            f"[{method_name}] "
            f"memory after T{task_index}: "
            f"{len(updated_memory):,}/"
            f"{strategy.memory_size:,}"
        )

    # ==================================================================
    # Final continual-learning summary
    # ==================================================================

    summary = build_summary(
        accuracy_matrix=accuracy_matrix
    )

    return {
        "method": method_name,
        "seed": seed,
        "memory_size": strategy.memory_size,
        "num_tasks": num_tasks,

        # Core CL evaluation.
        "accuracy_matrix": accuracy_matrix,

        # Complete training histories.
        "task_histories": task_histories,

        # Rich evaluation after every task.
        "evaluation": evaluation_history,

        # Average Accuracy + Average Forgetting.
        "summary": summary,
    }


# ============================================================================
# Output helpers
# ============================================================================


def save_method_results(
    result: Dict,
    output_dir: Path,
) -> None:
    """
    Save one replay method's complete results.
    """

    method_dir = (
        output_dir
        / result["method"].lower().replace(" ", "_")
    )

    method_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Complete machine-readable results.
    save_results(
        result,
        method_dir / "metrics.json",
    )

    # Compact summary.
    summary_path = method_dir / "summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result["summary"],
            handle,
            indent=2,
        )


def print_accuracy_matrix(
    matrix: Sequence[Sequence[float]],
) -> None:
    """
    Pretty-print the lower-triangular accuracy matrix.
    """

    print("\nAccuracy matrix:")

    header = "            " + " ".join(
        f"Test T{i}".rjust(10)
        for i in range(1, 6)
    )

    print(header)

    for row_index, row in enumerate(
        matrix,
        start=1,
    ):
        values = []

        for value in row:
            if value is None:
                values.append("".rjust(10))
            else:
                values.append(
                    f"{value:.3f}".rjust(10)
                )

        print(
            f"After T{row_index}".ljust(12)
            + " ".join(values)
        )


def print_final_metrics(
    result: Dict,
) -> None:
    """
    Print final T5 metrics for every task.
    """

    evaluation = result["evaluation"]

    final_results = evaluation.get("5", {})

    print("\nFinal T5 evaluation:")

    for task_id, metrics in final_results.items():
        print(
            f"  T{task_id}: "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RF-CL Random Replay and "
            "SNR-Aware Replay experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to experiment configuration.",
    )

    parser.add_argument(
        "--data",
        choices=["real"],
        default="real",
        help=(
            "Replay experiments currently require "
            "real RadioML data."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override training epochs from config."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for experiment outputs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training progress.",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    import yaml

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(config.get("seed", 42))
    )

    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(config["training"]["epochs"])
    )

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else config.get("output_dir", "results")
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"device={device}  "
        f"seed={seed}  "
        f"data={args.data}  "
        f"epochs={epochs}"
    )

    # ------------------------------------------------------------------
    # Global reproducibility
    # ------------------------------------------------------------------

    set_seed(seed)

    # ------------------------------------------------------------------
    # Load fixed RadioML task datasets
    # ------------------------------------------------------------------

    tasks = build_real_tasks(config)

    print("\nTask dataset sizes:")

    for task_id, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"  T{task_id}: "
            f"train={len(task['train']):,} "
            f"val={len(task['val']):,} "
            f"test={len(task['test']):,}"
        )

    memory_size = int(
        config["replay"]["memory_size"]
    )

    # ==================================================================
    # Random Replay
    # ==================================================================

    random_strategy = RandomReplay(
        memory_size=memory_size,
        seed=seed,
    )

    random_result = run_replay_method(
        "Random Replay",
        random_strategy,
        tasks,
        config,
        device=device,
        epochs=epochs,
        seed=seed,
        verbose=args.verbose,
    )

    # ==================================================================
    # SNR-Aware Replay
    # ==================================================================

    snr_strategy = SNRAwareReplay(
        memory_size=memory_size,
        seed=seed,
    )

    snr_result = run_replay_method(
        "SNR-Aware Replay",
        snr_strategy,
        tasks,
        config,
        device=device,
        epochs=epochs,
        seed=seed,
        verbose=args.verbose,
    )

    # ==================================================================
    # Print summaries
    # ==================================================================

    print("\n" + "=" * 72)
    print("RANDOM REPLAY")
    print("=" * 72)

    print_accuracy_matrix(
        random_result["accuracy_matrix"]
    )

    print(
        json.dumps(
            random_result["summary"],
            indent=2,
        )
    )

    print_final_metrics(random_result)

    print("\n" + "=" * 72)
    print("SNR-AWARE REPLAY")
    print("=" * 72)

    print_accuracy_matrix(
        snr_result["accuracy_matrix"]
    )

    print(
        json.dumps(
            snr_result["summary"],
            indent=2,
        )
    )

    print_final_metrics(snr_result)

    # ==================================================================
    # Save complete results
    # ==================================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_method_results(
        random_result,
        output_dir,
    )

    save_method_results(
        snr_result,
        output_dir,
    )

    # ------------------------------------------------------------------
    # Compact comparison file
    # ------------------------------------------------------------------

    combined = {
        "seed": seed,
        "epochs": epochs,
        "memory_size": memory_size,

        "random_replay": {
            **random_result["summary"],
        },

        "snr_aware_replay": {
            **snr_result["summary"],
        },
    }

    combined_path = (
        output_dir / "comparison.json"
    )

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            combined,
            handle,
            indent=2,
        )

    print(
        f"\nSaved replay results under: "
        f"{output_dir}/"
    )

    print("Done.")


if __name__ == "__main__":
    main()