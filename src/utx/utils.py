import math
import os
from typing import Optional

import torch


def get_device(device_pref: str = "cuda") -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CosineLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.base_lr * self.step_num / max(1, self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * min(1.0, progress)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location or "cpu")


