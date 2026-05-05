"""
Unified Conv1D forecaster used by daltoolboxdp via reticulate.

This module exposes a single Python file with a baseline class plus
specialized training behaviors selected through:
  - validation_strategy: "static" | "dynamic"
  - stopping_rule: "none" | "patience" | "sma" | "ema" | "h"
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, TensorDataset


VALIDATION_STRATEGIES = {"static", "dynamic"}
STOPPING_RULES = {"none", "patience", "sma", "ema", "h"}


@dataclass
class _TrainingConfig:
    n_epochs: int = 100
    lr: float = 0.001
    val_ratio: float = 0.2
    batch_size: int = 8
    patience: int = 100
    min_delta: float = 1e-4
    sma_window: int = 5
    ema_alpha: float = 0.2
    test_window: int = 30
    p_value: float = 0.05
    seed: Optional[int] = 42


class TsConv1DNet(nn.Module):
    def __init__(self, in_channels: int, input_dim: int):
        super().__init__()
        kernel_size = 1 if int(input_dim) == 1 else 2
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=int(in_channels), out_channels=64, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.rand(1, int(in_channels), int(input_dim))
            n_features = int(np.prod(self.feature_extractor(dummy).shape[1:]))
        self.regressor = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(n_features, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(x))


class _StopController:
    def __init__(self, rule: str, min_delta: float, patience: int, sma_window: int, ema_alpha: float, test_window: int, p_value: float):
        self.rule = rule
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

    def _clone_state(self, model: nn.Module):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _sma(self) -> float:
        return float(np.mean(self.val_history[-self.sma_window :]))

    def _ema(self, current: float) -> float:
        if self.ema_value is None:
            self.ema_value = current
        else:
            self.ema_value = self.ema_alpha * current + (1.0 - self.ema_alpha) * self.ema_value
        return float(self.ema_value)

    def _h_improved(self) -> bool:
        if len(self.val_history) < 2 * self.test_window:
            if len(self.val_history) == 1:
                return True
            return self.val_history[-1] < min(self.val_history[:-1])
        prev_window = self.val_history[-2 * self.test_window : -self.test_window]
        recent_window = self.val_history[-self.test_window :]
        _, p_value = ttest_ind(prev_window, recent_window, equal_var=False, alternative="greater")
        return bool(p_value < self.p_value)

    def step(self, model: nn.Module, current: float) -> bool:
        self.val_history.append(float(current))

        if self.rule == "none":
            self.best_state = self._clone_state(model)
            return False

        if self.rule == "h":
            improved = self._h_improved()
            if improved:
                self.best_state = self._clone_state(model)
                self.patience_ctr = 0
            else:
                self.patience_ctr += 1
            return self.patience_ctr >= self.patience

        if self.rule == "patience":
            monitor_value = float(current)
        elif self.rule == "sma":
            monitor_value = self._sma()
        elif self.rule == "ema":
            monitor_value = self._ema(float(current))
        else:
            raise ValueError(f"Unsupported stopping rule: {self.rule}")

        if (self.best_value - monitor_value) > self.min_delta:
            self.best_value = monitor_value
            self.best_state = self._clone_state(model)
            self.patience_ctr = 0
        else:
            self.patience_ctr += 1
        return self.patience_ctr >= self.patience


class TsConv1DModel:
    def __init__(self, in_channels: int, input_dim: int, validation_strategy: str = "static", stopping_rule: str = "none"):
        validation_strategy = str(validation_strategy).lower()
        stopping_rule = str(stopping_rule).lower()
        if validation_strategy not in VALIDATION_STRATEGIES:
            raise ValueError(f"validation_strategy must be one of {sorted(VALIDATION_STRATEGIES)}")
        if stopping_rule not in STOPPING_RULES:
            raise ValueError(f"stopping_rule must be one of {sorted(STOPPING_RULES)}")

        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.network = TsConv1DNet(int(in_channels), int(input_dim)).to(self._device())
        self.train_loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _prepare_xy(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        X = df.drop(columns=["t0"]).to_numpy().astype(np.float32)
        y = df["t0"].to_numpy().astype(np.float32)
        X = torch.from_numpy(X).unsqueeze(1)
        y = torch.from_numpy(y).unsqueeze(-1)
        return X, y

    @staticmethod
    def _make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(X, y), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    @staticmethod
    def _split_indices(n_samples: int, val_ratio: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        n_val = max(1, int(n_samples * float(val_ratio)))
        return idx[n_val:], idx[:n_val]

    def _epoch(self, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module) -> float:
        losses: List[float] = []
        if optimizer is None:
            self.network.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.float().to(self._device())
                    yb = yb.float().to(self._device())
                    losses.append(float(criterion(self.network(xb), yb).item()))
        else:
            self.network.train()
            for xb, yb in loader:
                xb = xb.float().to(self._device())
                yb = yb.float().to(self._device())
                optimizer.zero_grad()
                loss = criterion(self.network(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, df_train: pd.DataFrame, config: _TrainingConfig):
        if config.seed is not None:
            np.random.seed(int(config.seed))
            torch.manual_seed(int(config.seed))

        X_all, y_all = self._prepare_xy(df_train)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(config.lr))
        stopper = _StopController(
            rule=self.stopping_rule,
            min_delta=config.min_delta,
            patience=config.patience,
            sma_window=config.sma_window,
            ema_alpha=config.ema_alpha,
            test_window=config.test_window,
            p_value=config.p_value,
        )

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.epochs_done = 0

        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_idx, val_idx = self._split_indices(X_all.shape[0], config.val_ratio, config.seed)
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]
            train_loader = self._make_loader(X_train, y_train, config.batch_size, True)
            val_loader = self._make_loader(X_val, y_val, config.batch_size, False)
        elif self.validation_strategy == "static":
            train_loader = self._make_loader(X_all, y_all, config.batch_size, False)
            val_loader = None
        else:
            train_loader = None
            val_loader = None

        for epoch in range(int(config.n_epochs)):
            self.epochs_done += 1

            if self.validation_strategy == "dynamic":
                train_idx, val_idx = self._split_indices(X_all.shape[0], config.val_ratio, None if config.seed is None else int(config.seed) + epoch)
                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]
                train_loader = self._make_loader(X_train, y_train, config.batch_size, True)
                val_loader = self._make_loader(X_val, y_val, config.batch_size, False)

            train_loss = self._epoch(train_loader, optimizer, criterion)
            self.train_loss_hist.append(train_loss)

            if val_loader is not None:
                val_loss = self._epoch(val_loader, None, criterion)
                self.val_loss_hist.append(val_loss)
                if stopper.step(self.network, val_loss):
                    break

        if stopper.best_state is not None:
            self.network.load_state_dict(stopper.best_state)
        return self

    def predict(self, df_test: pd.DataFrame, batch_size: int = 8):
        X_test = df_test.drop(columns=["t0"], errors="ignore").to_numpy().astype(np.float32)
        X_test = torch.from_numpy(X_test).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_test, torch.zeros(X_test.shape[0], 1)), batch_size=int(batch_size), shuffle=False, drop_last=False)
        preds: List[torch.Tensor] = []
        self.network.eval()
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.network(xb.float().to(self._device())).detach().cpu())
        return torch.vstack(preds).squeeze(-1).numpy()


def ts_conv1d_create(in_channels, input_dim, validation_strategy="static", stopping_rule="none"):
    return TsConv1DModel(int(in_channels), int(input_dim), validation_strategy=validation_strategy, stopping_rule=stopping_rule)


def ts_conv1d_fit(
    model,
    df_train,
    n_epochs=100,
    lr=0.001,
    validation_strategy="static",
    stopping_rule="none",
    val_ratio=0.2,
    batch_size=8,
    patience=100,
    min_delta=1e-4,
    sma_window=5,
    ema_alpha=0.2,
    test_window=30,
    p_value=0.05,
    seed=42,
):
    model.validation_strategy = str(validation_strategy).lower()
    model.stopping_rule = str(stopping_rule).lower()
    config = _TrainingConfig(
        n_epochs=int(n_epochs),
        lr=float(lr),
        val_ratio=float(val_ratio),
        batch_size=int(batch_size),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
        seed=None if seed is None else int(seed),
    )
    return model.fit(df_train, config)


def ts_conv1d_predict(model, df_test, batch_size=8):
    return model.predict(df_test, batch_size=batch_size)
