"""
Unified variational autoencoder used by daltoolboxdp via reticulate.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, split_indices, validate_strategy


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_size), 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )
        self.mean_layer = nn.Linear(32, int(encoding_size))
        self.var_layer = nn.Linear(32, int(encoding_size))
        self.decoder = nn.Sequential(
            nn.Linear(int(encoding_size), 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, int(input_size)),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        hidden = self.encoder(x)
        return self.mean_layer(hidden), self.var_layer(hidden)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor):
        epsilon = torch.randn_like(var)
        return mean + var * epsilon

    def forward(self, x: torch.Tensor):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)
        return self.decode(z), mean, var


def _vae_loss(outputs, inputs, mean, var):
    reproduction_loss = nn.functional.binary_cross_entropy(outputs, inputs, reduction="sum")
    kld = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return reproduction_loss + kld


class VariationalAutoencoderModel:
    def __init__(self, input_size: int, encoding_size: int, validation_strategy: str = "static", stopping_rule: str = "none"):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.model = VariationalAutoencoder(input_size, encoding_size).float()
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _array(data):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool):
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor, tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    def _run_epoch(self, loader, optimizer: Optional[torch.optim.Optimizer]) -> float:
        losses: List[float] = []
        if optimizer is None:
            self.model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    out, mean, var = self.model(xb.float())
                    losses.append(float(_vae_loss(out, yb.float(), mean, var).item()))
        else:
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out, mean, var = self.model(xb.float())
                loss = _vae_loss(out, yb.float(), mean, var)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def fit(self, data, config: AutoencTrainingConfig):
        if config.seed is not None:
            np.random.seed(int(config.seed))
            torch.manual_seed(int(config.seed))
        array = self._array(data)
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
            self.train_loss.append(self._run_epoch(train_loader, optimizer))
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, None)
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
                _, mean, var = self.model(xb.float())
                outs.append(np.concatenate([mean.detach().numpy(), var.detach().numpy()], axis=1))
        return np.concatenate(outs, axis=0)

    def encode_decode(self, data, batch_size=32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                decoded, _, _ = self.model(xb.float())
                outs.append(decoded.detach().numpy())
        return np.concatenate(outs, axis=0)


def autoenc_variational_create(input_size, encoding_size, validation_strategy="static", stopping_rule="none"):
    return VariationalAutoencoderModel(input_size, encoding_size, validation_strategy=validation_strategy, stopping_rule=stopping_rule)


def autoenc_variational_fit(vae, data, batch_size=32, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05, seed=42):
    vae.validation_strategy, vae.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=vae.validation_strategy,
        stopping_rule=vae.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
        seed=None if seed is None else int(seed),
    )
    vae.fit(data, config)
    return vae, np.array(vae.train_loss), np.array(vae.val_loss)


def autoenc_variational_encode(vae, data, batch_size=32):
    return vae.encode(data, batch_size=batch_size)


def autoenc_variational_encode_decode(vae, data, batch_size=32):
    return vae.encode_decode(data, batch_size=batch_size)
