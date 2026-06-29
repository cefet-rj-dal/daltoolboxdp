"""
Unified convolutional autoencoder used by daltoolboxdp via reticulate.

The model follows the common Conv1D convention for univariate windows:
inputs are reshaped to (n_samples, 1, input_size), where the single channel
stores the observed series and input_size is the temporal length.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, split_indices, validate_strategy


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.input_size = int(input_size)
        hidden_channels = 16
        bottleneck_channels = 32
        self.encoder_features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.encoder_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels * self.input_size, int(encoding_size)),
        )
        self.decoder_projection = nn.Sequential(
            nn.Linear(int(encoding_size), bottleneck_channels * self.input_size),
            nn.LeakyReLU(),
            nn.Unflatten(1, (bottleneck_channels, self.input_size)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(bottleneck_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_projection(self.encoder_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        decoded = self.decoder_projection(latent)
        return self.decoder(decoded)


class ConvAutoencoderModel:
    def __init__(self, input_size: int, encoding_size: int, validation_strategy: str = "static", stopping_rule: str = "none"):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.input_size = int(input_size)
        self.model = ConvAutoencoder(self.input_size, encoding_size).float()
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    def _array(self, data):
        if isinstance(data, pd.DataFrame):
            array = data.to_numpy().astype(np.float32)
        else:
            array = np.asarray(data, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Conv autoencoder expects a 2D array with shape (n_samples, input_size).")
        if array.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {array.shape[1]}.")
        return array[:, np.newaxis, :]

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
        array = self._array(data)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.learning_rate))
        stopper = StopController(self.stopping_rule, config.min_delta, config.patience, config.sma_window, config.ema_alpha, config.test_window, config.p_value)
        self.train_loss = []
        self.val_loss = []
        self.epochs_done = 0

        if self.validation_strategy == "static" and self.stopping_rule != "none":
            train_idx, val_idx = split_indices(array.shape[0], config.val_ratio)
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
                train_idx, val_idx = split_indices(array.shape[0], config.val_ratio)
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

    def encode(self, data, batch_size=32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(self.model.encode(xb.float()).detach().numpy())
        return np.concatenate(outs, axis=0)

    def encode_decode(self, data, batch_size=32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(self.model(xb.float()).detach().numpy().squeeze(1))
        return np.concatenate(outs, axis=0)


def autoenc_conv_create(input_size, encoding_size, validation_strategy="static", stopping_rule="none"):
    return ConvAutoencoderModel(input_size, encoding_size, validation_strategy=validation_strategy, stopping_rule=stopping_rule)


def autoenc_conv_fit(cae, data, batch_size=32, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05):
    cae.validation_strategy, cae.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=cae.validation_strategy,
        stopping_rule=cae.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
    )
    cae.fit(data, config)
    return cae, np.array(cae.train_loss), np.array(cae.val_loss)


def autoenc_conv_encode(cae, data, batch_size=32):
    return cae.encode(data, batch_size=batch_size)


def autoenc_conv_encode_decode(cae, data, batch_size=32):
    return cae.encode_decode(data, batch_size=batch_size)
