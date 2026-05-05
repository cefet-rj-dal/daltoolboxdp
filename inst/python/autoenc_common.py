"""
Shared training utilities for daltoolboxdp autoencoders.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import ttest_ind


VALIDATION_STRATEGIES = {"static", "dynamic"}
STOPPING_RULES = {"none", "patience", "sma", "ema", "h"}


@dataclass
class AutoencTrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    validation_strategy: str = "static"
    stopping_rule: str = "none"
    val_ratio: float = 0.3
    patience: int = 100
    min_delta: float = 1e-4
    sma_window: int = 5
    ema_alpha: float = 0.2
    test_window: int = 30
    p_value: float = 0.05
    seed: Optional[int] = 42


def validate_strategy(validation_strategy: str, stopping_rule: str) -> Tuple[str, str]:
    validation_strategy = str(validation_strategy).lower()
    stopping_rule = str(stopping_rule).lower()
    if validation_strategy not in VALIDATION_STRATEGIES:
        raise ValueError(f"validation_strategy must be one of {sorted(VALIDATION_STRATEGIES)}")
    if stopping_rule not in STOPPING_RULES:
        raise ValueError(f"stopping_rule must be one of {sorted(STOPPING_RULES)}")
    return validation_strategy, stopping_rule


def split_indices(n_samples: int, val_ratio: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_val = max(1, int(n_samples * float(val_ratio)))
    return idx[n_val:], idx[:n_val]


class StopController:
    def __init__(self, rule: str, min_delta: float, patience: int, sma_window: int, ema_alpha: float, test_window: int, p_value: float):
        self.rule = str(rule).lower()
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.sma_window = max(1, int(sma_window))
        self.ema_alpha = float(ema_alpha)
        self.test_window = max(2, int(test_window))
        self.p_value = float(p_value)
        self.best_value = float("inf")
        self.best_state = None
        self.patience_ctr = 0
        self.ema_value = None
        self.val_history: List[float] = []

    def clone_state(self, model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _h_improved(self) -> bool:
        if len(self.val_history) < 2 * self.test_window:
            if len(self.val_history) == 1:
                return True
            return self.val_history[-1] < min(self.val_history[:-1])
        previous = self.val_history[-2 * self.test_window : -self.test_window]
        recent = self.val_history[-self.test_window :]
        _, p_value = ttest_ind(previous, recent, equal_var=False, alternative="greater")
        return bool(p_value < self.p_value)

    def step(self, model, current: float) -> bool:
        self.val_history.append(float(current))

        if self.rule == "none":
            self.best_state = self.clone_state(model)
            return False

        if self.rule == "h":
            improved = self._h_improved()
            if improved:
                self.best_state = self.clone_state(model)
                self.patience_ctr = 0
            else:
                self.patience_ctr += 1
            return self.patience_ctr >= self.patience

        if self.rule == "patience":
            monitor_value = float(current)
        elif self.rule == "sma":
            monitor_value = float(np.mean(self.val_history[-self.sma_window :]))
        elif self.rule == "ema":
            if self.ema_value is None:
                self.ema_value = current
            else:
                self.ema_value = self.ema_alpha * current + (1.0 - self.ema_alpha) * self.ema_value
            monitor_value = float(self.ema_value)
        else:
            raise ValueError(f"Unsupported stopping rule: {self.rule}")

        if (self.best_value - monitor_value) > self.min_delta:
            self.best_value = monitor_value
            self.best_state = self.clone_state(model)
            self.patience_ctr = 0
        else:
            self.patience_ctr += 1
        return self.patience_ctr >= self.patience
