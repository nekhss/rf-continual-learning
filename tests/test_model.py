"""Unit tests for Person 2's deliverables: model, trainer, and baselines.

Run from the repo root:  pytest tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make 'src' importable when pytest is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn import RFClassifier, build_model
from src.training.seed import set_seed
from src.training.trainer import (
    Trainer, build_loader, create_criterion, create_optimizer, evaluate, fit,
    train_one_epoch, unpack_batch,
)
from src.training.synthetic import make_synthetic_tasks, RFDictDataset
from src.continual.naive import run_naive_sequential
from src.continual.joint import run_joint_training

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_forward_output_shape():
    model = RFClassifier(num_classes=11)
    x = torch.randn(8, 2, 128)
    logits = model(x)
    assert logits.shape == (8, 11)


def test_forward_accepts_various_batch_sizes():
    model = RFClassifier(num_classes=11).eval()
    for b in (1, 3, 16):
        assert model(torch.randn(b, 2, 128)).shape == (b, 11)


def test_num_classes_configurable():
    model = RFClassifier(num_classes=5)
    assert model(torch.randn(4, 2, 128)).shape == (4, 5)


def test_build_model_from_config():
    cfg = {"num_classes": 11,
           "model": {"in_channels": 2, "conv_channels": [16, 32],
                     "kernel_size": 5, "fc_hidden": 0, "dropout": 0.0}}
    model = build_model(cfg)
    assert model(torch.randn(2, 2, 128)).shape == (2, 11)
    assert model.num_parameters() > 0


def test_rejects_wrong_channel_count():
    model = RFClassifier(num_classes=11)
    try:
        model(torch.randn(4, 3, 128))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_backward_populates_gradients():
    model = RFClassifier(num_classes=11)
    x = torch.randn(4, 2, 128)
    y = torch.randint(0, 11, (4,))
    loss = create_criterion()(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_loss_is_finite_scalar():
    model = RFClassifier(num_classes=11)
    logits = model(torch.randn(6, 2, 128))
    loss = create_criterion()(logits, torch.randint(0, 11, (6,)))
    assert loss.dim() == 0 and torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Trainer plumbing
# --------------------------------------------------------------------------- #
def test_unpack_batch_dict_and_tuple():
    x = torch.randn(2, 2, 128)
    y = torch.tensor([0, 1])
    dx, dy = unpack_batch({"x": x, "y": y, "snr": torch.tensor([0, 0]), "task": torch.tensor([0, 0])})
    tx, ty = unpack_batch((x, y))
    assert torch.equal(dx, x) and torch.equal(dy, y)
    assert torch.equal(tx, x) and torch.equal(ty, y)


def test_optimizer_factory_variants():
    model = RFClassifier(num_classes=11)
    for name in ("adam", "adamw", "sgd"):
        opt = create_optimizer(model, {"training": {"optimizer": name, "learning_rate": 1e-3}})
        assert len(opt.param_groups) >= 1


def test_train_one_epoch_runs_on_dict_batches():
    set_seed(42)
    ds = make_synthetic_tasks(num_tasks=1, train_per_task=64, val_per_task=8,
                              test_per_task=8, seed=0)[0]["train"]
    loader = build_loader(ds, {"training": {"batch_size": 16}}, shuffle=True, seed=0)
    model = RFClassifier(num_classes=11)
    opt = create_optimizer(model, {"training": {"optimizer": "adam", "learning_rate": 1e-3}})
    stats = train_one_epoch(model, loader, opt, create_criterion(), DEVICE)
    assert np.isfinite(stats["loss"]) and 0.0 <= stats["acc"] <= 1.0


def test_tiny_training_run_reduces_loss():
    """A short real training loop on synthetic data should lower the loss."""
    set_seed(42)
    task = make_synthetic_tasks(num_tasks=1, train_per_task=256, val_per_task=64,
                                test_per_task=64, seed=1)[0]
    cfg = {"num_classes": 11,
           "training": {"batch_size": 32, "epochs": 4, "optimizer": "adam",
                        "learning_rate": 1e-3, "early_stopping_patience": 0}}
    train = build_loader(task["train"], cfg, shuffle=True, seed=1)
    val = build_loader(task["val"], cfg, shuffle=False, seed=1)
    model = build_model(cfg)
    hist = fit(model, train, val, config=cfg, device=DEVICE)
    assert len(hist["train_loss"]) == 4
    assert hist["train_loss"][-1] < hist["train_loss"][0]  # learning happened


# --------------------------------------------------------------------------- #
# Baselines (smoke-level, using tiny synthetic tasks)
# --------------------------------------------------------------------------- #
def _tiny_task_loaders(num_tasks=3, seed=42):
    cfg = {"training": {"batch_size": 32}}
    tasks = make_synthetic_tasks(num_tasks=num_tasks, train_per_task=96,
                                 val_per_task=32, test_per_task=32, seed=seed)
    return [
        {"train": build_loader(t["train"], cfg, shuffle=True, seed=seed),
         "val": build_loader(t["val"], cfg, shuffle=False, seed=seed),
         "test": build_loader(t["test"], cfg, shuffle=False, seed=seed)}
        for t in tasks
    ], cfg


def test_naive_matrix_is_lower_triangular():
    set_seed(42)
    loaders, base_cfg = _tiny_task_loaders(num_tasks=3)
    cfg = {"num_classes": 11,
           "training": {**base_cfg["training"], "epochs": 1,
                        "optimizer": "adam", "learning_rate": 1e-3,
                        "early_stopping_patience": 0}}
    model = build_model(cfg)
    res = run_naive_sequential(model, loaders, cfg, device=DEVICE)
    m = res["accuracy_matrix"]
    assert m.shape == (3, 3)
    # lower triangle (incl. diagonal) filled; upper triangle NaN
    for i in range(3):
        for j in range(3):
            if j <= i:
                assert np.isfinite(m[i, j])
            else:
                assert np.isnan(m[i, j])
    assert len(res["final_accuracies"]) == 3


def test_joint_returns_per_task_accuracy():
    set_seed(42)
    loaders, base_cfg = _tiny_task_loaders(num_tasks=3)
    cfg = {"num_classes": 11,
           "training": {**base_cfg["training"], "epochs": 1,
                        "optimizer": "adam", "learning_rate": 1e-3,
                        "early_stopping_patience": 0}}
    model = build_model(cfg)
    res = run_joint_training(model, loaders, cfg, device=DEVICE, seed=42)
    assert len(res["per_task_accuracy"]) == 3
    assert all(0.0 <= a <= 1.0 for a in res["per_task_accuracy"])
    assert np.isfinite(res["mean_accuracy"])


def test_dataset_matches_contract():
    ds = make_synthetic_tasks(num_tasks=1, train_per_task=4, val_per_task=2,
                              test_per_task=2, seed=0)[0]["train"]
    sample = ds[0]
    assert set(sample.keys()) == {"x", "y", "snr", "task"}
    assert tuple(sample["x"].shape) == (2, 128)
    assert isinstance(sample["y"], int)
