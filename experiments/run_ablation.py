from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.training.seed import set_seed
from src.evaluation.evaluate import save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-CL replay ablation experiments."
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to experiment configuration.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs from config.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/ablation",
        help="Directory for ablation outputs.",
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


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("RF-CL ABLATION EXPERIMENT")
    print("=" * 72)
    print(f"device={device}")
    print(f"seed={seed}")
    print(f"epochs={epochs}")

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------

    set_seed(seed)

    # ---------------------------------------------------------------
    # Import the already-tested experiment runners
    # ---------------------------------------------------------------

    from experiments.run_baselines import (
        build_real_tasks,
    )

    from experiments.run_replay import (
        run_replay_method,
    )

    from src.continual.replay import (
        RandomReplay,
        SNRAwareReplay,
    )

    # ---------------------------------------------------------------
    # Load the SAME fixed real-data task datasets
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # 1. RANDOM REPLAY
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ABLATION 1: RANDOM REPLAY")
    print("=" * 72)

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

    # ---------------------------------------------------------------
    # 2. SNR-AWARE REPLAY
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ABLATION 2: SNR-AWARE REPLAY")
    print("=" * 72)

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

    # ---------------------------------------------------------------
    # Extract final metrics
    # ---------------------------------------------------------------

    random_summary = random_result["summary"]
    snr_summary = snr_result["summary"]

    aa_difference = (
        snr_summary["average_accuracy"]
        - random_summary["average_accuracy"]
    )

    af_difference = (
        random_summary["average_forgetting"]
        - snr_summary["average_forgetting"]
    )

    # ---------------------------------------------------------------
    # Save complete individual results
    # ---------------------------------------------------------------

    save_results(
        random_result,
        output_dir / "random_replay_metrics.json",
    )

    save_results(
        snr_result,
        output_dir / "snr_aware_replay_metrics.json",
    )

    # ---------------------------------------------------------------
    # Compact ablation comparison
    # ---------------------------------------------------------------

    comparison = {
        "experiment": "RF-CL Replay Ablation",
        "seed": seed,
        "epochs": epochs,
        "memory_size": memory_size,

        "methods": {
            "random_replay": {
                "average_accuracy": (
                    random_summary["average_accuracy"]
                ),
                "average_forgetting": (
                    random_summary["average_forgetting"]
                ),
            },
            "snr_aware_replay": {
                "average_accuracy": (
                    snr_summary["average_accuracy"]
                ),
                "average_forgetting": (
                    snr_summary["average_forgetting"]
                ),
            },
        },

        "comparison": {
            "snr_aware_minus_random_average_accuracy": (
                aa_difference
            ),
            "random_minus_snr_aware_average_forgetting": (
                af_difference
            ),
        },
    }

    comparison_path = (
        output_dir / "comparison.json"
    )

    with comparison_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            comparison,
            handle,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Print final comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)

    print(
        f"Random Replay:"
        f"  AA={random_summary['average_accuracy']:.4f}"
        f"  AF={random_summary['average_forgetting']:.4f}"
    )

    print(
        f"SNR-Aware Replay:"
        f"  AA={snr_summary['average_accuracy']:.4f}"
        f"  AF={snr_summary['average_forgetting']:.4f}"
    )

    print(
        "\nDifference:"
    )

    print(
        f"  SNR-Aware AA improvement: "
        f"{aa_difference:+.4f}"
    )

    print(
        f"  SNR-Aware forgetting reduction: "
        f"{af_difference:+.4f}"
    )

    print(
        f"\nSaved ablation results under: "
        f"{output_dir}/"
    )

    print("Done.")


if __name__ == "__main__":
    main()