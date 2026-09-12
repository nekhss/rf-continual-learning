"""
Public data-pipeline API for RF-CL.

Other modules (models / continual-learning / training / evaluation) should
only ever need:

    from src.data import get_task_dataset, get_task_dataloader

Everything else in this package is an implementation detail.
"""

from src.data.dataset import get_task_dataset, get_task_dataloader, RFModDataset
from src.data.tasks import TASK_SNR_RANGES, NUM_TASKS

__all__ = [
    "get_task_dataset",
    "get_task_dataloader",
    "RFModDataset",
    "TASK_SNR_RANGES",
    "NUM_TASKS",
]
