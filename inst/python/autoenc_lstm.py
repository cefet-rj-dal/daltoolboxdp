"""
Unified LSTM autoencoder used by daltoolboxdp via reticulate.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, split_indices, validate_strategy


class Encoder(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=int(input_size), hidden_size=int(encoding_size), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1].unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=int(encoding_size), hidden_size=int(encoding_size), batch_first=True)
        self.output_layer = nn.Linear(int(encoding_size), int(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.output_layer(out).view(out.size(0), 1, -1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.encoder = Encoder(input_size, encoding_size)
        self.decoder = Decoder(input_size, encoding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class LSTMAutoencoderModel:
    def __init__(self, input_size: int, encoding_size: int, validation_strategy: str = "static", stopping_rule: str = "none"):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.model = LSTMAutoencoder(input_size, encoding_size).float()
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _array(data):
        if isinstance(data, pd.DataFrame):
            array = data.to_numpy().astype(np.float32)
        else:
            array = np.asarray(data, dtype=np.float32)
        return array.reshape(array.shape[0], 1, array.shape[1])

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool):
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor, tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    def _run_epoch(self, loader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module) -> float:
        losses: List[float] = []
        if optimizer is None:
            self.model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    losses.append(float(criterion(self.model(xb.float()), yb.float()).item()))
        else:
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb.float()), yb.float())
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, data, config: AutoencTrainingConfig):
        if config.seed is not None:
            np.random.seed(int(config.seed))
            torch.manual_seed(int(config.seed))
        array = self._array(data)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.learning_rate))
        stopper = StopController(self.stopping_rule, config.min_delta, config.patience, config.sma_window, config.ema_alpha, config.test_window, config.p_value)
        self.train_loss = []
        self.val_loss = []
        self.epochs_done = 0

        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_idx, val_idx = split_indices(array.shape[0], config.val_ratio, config.seed)
            train_loader = self._loader(array[train_idx], config.batch_size, True)
            val_loader = self._loader(array[val_idx], config.batch_size, False)
        elif self.validation_strategy == "static":
            train_loader = self._loader(array, config.batch_size, True)
            val_loader = None
        else:
            train_loader = None
            val_loader = None

        for epoch in range(int(config.num_epochs)):
            self.epochs_done += 1
            if self.validation_strategy == "dynamic":
                train_idx, val_idx = split_indices(array.shape[0], config.val_ratio, None if config.seed is None else int(config.seed) + epoch)
                train_loader = self._loader(array[train_idx], config.batch_size, True)
                val_loader = self._loader(array[val_idx], config.batch_size, False)
            self.train_loss.append(self._run_epoch(train_loader, optimizer, criterion))
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, None, criterion)
                self.val_loss.append(val_loss)
                if stopper.step(self.model, val_loss):
                    break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        return self

    def encode(self, data, batch_size=20):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(self.model.encoder(xb.float()).detach().numpy().reshape(xb.size(0), -1))
        return np.concatenate(outs, axis=0)

    def encode_decode(self, data, batch_size=20):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(self.model(xb.float()).detach().numpy())
        return np.concatenate(outs, axis=0)


def autoenc_lstm_create(input_size, encoding_size, validation_strategy="static", stopping_rule="none"):
    return LSTMAutoencoderModel(input_size, encoding_size, validation_strategy=validation_strategy, stopping_rule=stopping_rule)


def autoenc_lstm_fit(lae, data, batch_size=20, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05, seed=42, return_loss=False):
    lae.validation_strategy, lae.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=lae.validation_strategy,
        stopping_rule=lae.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
        seed=None if seed is None else int(seed),
    )
    lae.fit(data, config)
    return lae, np.array(lae.train_loss), np.array(lae.val_loss)


def autoenc_lstm_encode(lae, data, batch_size=20):
    return lae.encode(data, batch_size=batch_size)


def autoenc_lstm_encode_decode(lae, data, batch_size=20):
    return lae.encode_decode(data, batch_size=batch_size)
