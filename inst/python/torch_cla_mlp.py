"""
Unified PyTorch MLP classifier used by daltoolboxdp via reticulate.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, TensorDataset


VALIDATION_STRATEGIES = {"static", "dynamic"}
STOPPING_RULES = {"none", "patience", "sma", "ema", "h"}


@dataclass
class _TrainingConfig:
    epochs: int = 100
    lr: float = 0.001
    val_ratio: float = 0.2
    batch_size: int = 64
    patience: int = 100
    min_delta: float = 1e-4
    sma_window: int = 5
    ema_alpha: float = 0.2
    test_window: int = 30
    p_value: float = 0.05
    weight_decay: float = 0.0


def _activation_module(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _normalization_module(kind: str, dim: int) -> Optional[nn.Module]:
    kind = str(kind).lower()
    if kind == "none":
        return None
    if kind == "batch":
        return nn.BatchNorm1d(int(dim))
    if kind == "layer":
        return nn.LayerNorm(int(dim))
    raise ValueError("normalization must be one of {'none', 'batch', 'layer'}")


def _apply_init(module: nn.Module, init_method: str):
    init_method = str(init_method).lower()
    if isinstance(module, nn.Linear):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif init_method == "default":
            return
        else:
            raise ValueError(f"Unsupported init_method: {init_method}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TorchMLPClassifierNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "none",
        init_method: str = "default",
    ):
        super().__init__()
        layers = []
        prev = int(input_dim)
        if isinstance(hidden_sizes, (int, np.integer)):
            hidden_sizes = [int(hidden_sizes)]
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            norm = _normalization_module(normalization, int(h))
            if norm is not None:
                layers.append(norm)
            layers.append(_activation_module(activation))
            if float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda module: _apply_init(module, init_method))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
            self.best_state = self._clone_state(model)
            self.patience_ctr = 0
        else:
            self.patience_ctr += 1
        return self.patience_ctr >= self.patience


class TorchMLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "none",
        init_method: str = "default",
        validation_strategy: str = "static",
        stopping_rule: str = "none",
    ):
        validation_strategy = str(validation_strategy).lower()
        stopping_rule = str(stopping_rule).lower()
        if validation_strategy not in VALIDATION_STRATEGIES:
            raise ValueError(f"validation_strategy must be one of {sorted(VALIDATION_STRATEGIES)}")
        if stopping_rule not in STOPPING_RULES:
            raise ValueError(f"stopping_rule must be one of {sorted(STOPPING_RULES)}")
        self.validation_strategy = validation_strategy
        self.stopping_rule = stopping_rule
        self.network = TorchMLPClassifierNet(
            input_dim,
            hidden_sizes,
            num_classes,
            dropout=dropout,
            activation=activation,
            normalization=normalization,
            init_method=init_method,
        ).to(self._device())
        self.classes_: List = []
        self.train_loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _prepare_xy(df: pd.DataFrame, target_column: str, classes_: Optional[List]) -> Tuple[torch.Tensor, torch.Tensor, List]:
        X = df.drop(columns=[target_column]).to_numpy().astype(np.float32)
        y_raw = df[target_column].to_numpy()
        if classes_ is None:
            classes_ = sorted(pd.Series(y_raw).astype("category").cat.categories.tolist())
        class_to_idx = {c: i for i, c in enumerate(classes_)}
        y = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(y), classes_

    @staticmethod
    def _split_indices(n_samples: int, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        n_val = max(1, int(n_samples * float(val_ratio)))
        return idx[n_val:], idx[:n_val]

    def _epoch(self, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module) -> float:
        losses: List[float] = []
        if optimizer is None:
            self.network.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.float().to(self._device())
                    yb = yb.long().to(self._device())
                    losses.append(float(criterion(self.network(xb), yb).item()))
        else:
            self.network.train()
            for xb, yb in loader:
                xb = xb.float().to(self._device())
                yb = yb.long().to(self._device())
                optimizer.zero_grad()
                loss = criterion(self.network(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, df_train: pd.DataFrame, target_column: str, config: _TrainingConfig, classes_: Optional[List] = None):

        X_all, y_all, self.classes_ = self._prepare_xy(df_train, target_column, classes_)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
        stopper = _StopController(self.stopping_rule, config.min_delta, config.patience, config.sma_window, config.ema_alpha, config.test_window, config.p_value)

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.epochs_done = 0

        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_idx, val_idx = self._split_indices(X_all.shape[0], config.val_ratio)
            train_loader = DataLoader(TensorDataset(X_all[train_idx], y_all[train_idx]), batch_size=int(config.batch_size), shuffle=True, drop_last=False)
            val_loader = DataLoader(TensorDataset(X_all[val_idx], y_all[val_idx]), batch_size=int(config.batch_size), shuffle=False, drop_last=False)
        elif self.validation_strategy == "static":
            train_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=int(config.batch_size), shuffle=True, drop_last=False)
            val_loader = None
        else:
            train_loader = None
            val_loader = None

        for epoch in range(int(config.epochs)):
            self.epochs_done += 1
            if self.validation_strategy == "dynamic":
                train_idx, val_idx = self._split_indices(X_all.shape[0], config.val_ratio)
                train_loader = DataLoader(TensorDataset(X_all[train_idx], y_all[train_idx]), batch_size=int(config.batch_size), shuffle=True, drop_last=False)
                val_loader = DataLoader(TensorDataset(X_all[val_idx], y_all[val_idx]), batch_size=int(config.batch_size), shuffle=False, drop_last=False)

            self.train_loss_hist.append(self._epoch(train_loader, optimizer, criterion))
            if val_loader is not None:
                val_loss = self._epoch(val_loader, None, criterion)
                self.val_loss_hist.append(val_loss)
                if stopper.step(self.network, val_loss):
                    break

        if stopper.best_state is not None:
            self.network.load_state_dict(stopper.best_state)
        return self

    def predict(self, df_test: pd.DataFrame):
        X = torch.from_numpy(df_test.to_numpy().astype(np.float32))
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self._device()))
            pred_idx = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        idx_to_class = {i: c for i, c in enumerate(self.classes_)}
        return [idx_to_class[i] for i in pred_idx]

    def predict_scores(self, df_test: pd.DataFrame):
        X = torch.from_numpy(df_test.to_numpy().astype(np.float32))
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self._device()))
            scores = F.softmax(logits, dim=-1).cpu().numpy()
        return scores.tolist()


def torch_cla_mlp_create(
    input_dim: int,
    hidden_sizes: List[int],
    num_classes: int,
    dropout: float = 0.0,
    activation: str = "relu",
    normalization: str = "none",
    init_method: str = "default",
    validation_strategy: str = "static",
    stopping_rule: str = "none",
):
    return TorchMLPClassifier(
        input_dim,
        hidden_sizes,
        num_classes,
        dropout=dropout,
        activation=activation,
        normalization=normalization,
        init_method=init_method,
        validation_strategy=validation_strategy,
        stopping_rule=stopping_rule,
    )


def torch_cla_mlp_fit(
    model,
    df_train: pd.DataFrame,
    target_column: str,
    epochs: int = 100,
    lr: float = 1e-3,
    validation_strategy: str = "static",
    stopping_rule: str = "none",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    patience: int = 100,
    min_delta: float = 1e-4,
    sma_window: int = 5,
    ema_alpha: float = 0.2,
    test_window: int = 30,
    p_value: float = 0.05,
    weight_decay: float = 0.0,
    classes_: Optional[List] = None,
):
    model.validation_strategy = str(validation_strategy).lower()
    model.stopping_rule = str(stopping_rule).lower()
    config = _TrainingConfig(
        epochs=int(epochs),
        lr=float(lr),
        val_ratio=float(val_ratio),
        batch_size=int(batch_size),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
        weight_decay=float(weight_decay),
    )
    return model.fit(df_train, target_column=target_column, config=config, classes_=classes_)


def torch_cla_mlp_predict(model, df_test: pd.DataFrame, classes_: Optional[List] = None):
    if classes_ is not None and not model.classes_:
        model.classes_ = list(classes_)
    return model.predict(df_test)


def torch_cla_mlp_predict_scores(model, df_test: pd.DataFrame, classes_: Optional[List] = None):
    if classes_ is not None and not model.classes_:
        model.classes_ = list(classes_)
    return model.predict_scores(df_test)
