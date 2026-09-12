"""Generic training infrastructure.

This module is deliberately *continual-learning agnostic*. It knows how to:
  * create optimizers / loss,
  * build DataLoaders deterministically,
  * run one epoch, validate, and fit for N epochs (with optional early stopping),
  * save / load checkpoints.

It does NOT know about tasks, replay, EWC, LwF, etc. Baselines and CL methods
compose these primitives.

Batch format
------------
Batches may be either the agreed dataset dict ``{"x", "y", "snr", "task"}`` or a
plain ``(x, y)`` tuple. Only ``x`` and ``y`` are ever used here.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .seed import make_generator, seed_worker


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return a torch device (CUDA if available unless overridden)."""
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _training_cfg(config: Optional[dict]) -> dict:
    """Extract the ``training`` sub-dict, tolerating a flat dict too."""
    if not config:
        return {}
    if "training" in config and isinstance(config["training"], dict):
        return config["training"]
    return config


def unpack_batch(batch: Any):
    """Return ``(x, y)`` from a dict batch or a (x, y[, ...]) sequence."""
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        "Unsupported batch type. Expected dict with keys 'x'/'y' or a (x, y) "
        f"sequence, got: {type(batch)!r}"
    )


def build_loader(
    dataset: Dataset,
    config: Optional[dict] = None,
    *,
    shuffle: bool = False,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with deterministic shuffling.

    Uses a seeded generator so that, given the same seed, shuffling order is
    reproducible across baselines.
    """
    tcfg = _training_cfg(config)
    bs = int(batch_size if batch_size is not None else tcfg.get("batch_size", 128))
    num_workers = int(tcfg.get("num_workers", 0))
    kwargs: Dict[str, Any] = dict(
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    if shuffle:
        kwargs["generator"] = make_generator(seed)
    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **kwargs)


# --------------------------------------------------------------------------- #
# Optimizer / loss
# --------------------------------------------------------------------------- #
def create_optimizer(model: nn.Module, config: Optional[dict] = None) -> torch.optim.Optimizer:
    tcfg = _training_cfg(config)
    name = str(tcfg.get("optimizer", "adam")).lower()
    lr = float(tcfg.get("learning_rate", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(tcfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Use adam | adamw | sgd.")


def create_criterion(config: Optional[dict] = None) -> nn.Module:
    """Classification loss. Kept as a function so it is easy to swap later."""
    return nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        x, y = unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += bs
    seen = max(seen, 1)
    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    """Compute loss/accuracy on a loader. Used for validation AND test reporting.

    NOTE: passing a *test* loader here only reads it for reporting -- it never
    influences training (no gradients, no early-stopping decisions).
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        x, y = unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.size(0)
    seen = max(seen, 1)
    return {"loss": total_loss / seen, "acc": correct / seen}


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Convenience wrapper returning top-1 accuracy as a fraction in [0, 1]."""
    return evaluate(model, loader, criterion=None, device=device)["acc"]


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    epochs: Optional[int] = None,
    verbose: bool = False,
    log_prefix: str = "",
) -> Dict[str, Any]:
    """Train ``model`` for N epochs, optionally early-stopping on validation loss.

    Returns a history dict with per-epoch metrics plus ``best_epoch``. When a
    validation loader is given, the best (lowest val-loss) weights are restored
    into ``model`` before returning.

    Early stopping uses VALIDATION only -- never test data -- so it is a legal
    hyperparameter/stopping signal.
    """
    tcfg = _training_cfg(config)
    device = device or get_device()
    model.to(device)

    optimizer = create_optimizer(model, config)
    criterion = create_criterion(config)
    epochs = int(epochs if epochs is not None else tcfg.get("epochs", 50))
    patience = int(tcfg.get("early_stopping_patience", 0))
    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))

    history: Dict[str, Any] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "best_epoch": epochs,
    }

    best_val = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])

        msg = f"{log_prefix}epoch {epoch:3d}/{epochs}  train_loss={tr['loss']:.4f} acc={tr['acc']:.3f}"

        if val_loader is not None:
            va = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["acc"])
            msg += f"  val_loss={va['loss']:.4f} acc={va['acc']:.3f}"

            if va["loss"] < best_val - 1e-6:
                best_val = va["loss"]
                best_state = _clone_state(model)
                history["best_epoch"] = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if verbose:
            print(msg, flush=True)

        if val_loader is not None and patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {epoch} "
                      f"(best epoch {history['best_epoch']})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[dict] = None,
) -> None:
    payload: Dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Any] = None,
) -> dict:
    payload = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("extra", {})


class Trainer:
    """Thin object wrapper over the functions above for callers who prefer it."""

    def __init__(self, config: Optional[dict] = None, device: Optional[torch.device] = None):
        self.config = config or {}
        self.device = device or get_device()

    def fit(self, model, train_loader, val_loader=None, **kwargs):
        return fit(model, train_loader, val_loader, config=self.config,
                   device=self.device, **kwargs)

    def evaluate(self, model, loader):
        return evaluate(model, loader, create_criterion(self.config), self.device)

    def accuracy(self, model, loader):
        return accuracy(model, loader, self.device)
