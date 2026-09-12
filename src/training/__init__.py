"""Generic training infrastructure owned by Person 2."""

from .seed import set_seed, make_generator, seed_worker
from .trainer import (
    Trainer,
    fit,
    train_one_epoch,
    evaluate,
    accuracy,
    create_optimizer,
    create_criterion,
    build_loader,
    unpack_batch,
    get_device,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "set_seed", "make_generator", "seed_worker",
    "Trainer", "fit", "train_one_epoch", "evaluate", "accuracy",
    "create_optimizer", "create_criterion", "build_loader", "unpack_batch",
    "get_device", "save_checkpoint", "load_checkpoint",
]
