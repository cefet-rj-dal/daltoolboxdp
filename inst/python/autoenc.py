"""
Unified dense autoencoder used by daltoolboxdp via reticulate.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, build_dense_stack, split_indices, validate_strategy


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes=None,
        decoder_hidden_sizes=None,
        activation: str = "relu",
        output_activation: str = "none",
        negative_slope: float = 0.2,
    ):
        super().__init__()
        encoder_hidden_sizes = [64] if encoder_hidden_sizes is None else encoder_hidden_sizes
        decoder_hidden_sizes = list(reversed(list(encoder_hidden_sizes))) if decoder_hidden_sizes is None else decoder_hidden_sizes
        self.encoder = build_dense_stack(
            int(input_size),
            encoder_hidden_sizes,
            int(encoding_size),
            activation=activation,
            negative_slope=negative_slope,
        )
        self.decoder = build_dense_stack(
            int(encoding_size),
            decoder_hidden_sizes,
            int(input_size),
            activation=activation,
            output_activation=output_activation,
            negative_slope=negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DenseAutoencoderModel:
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes=None,
        decoder_hidden_sizes=None,
        activation: str = "relu",
        output_activation: str = "none",
        negative_slope: float = 0.2,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
    ):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.model = Autoencoder(
            int(input_size),
            int(encoding_size),
            encoder_hidden_sizes=encoder_hidden_sizes,
            decoder_hidden_sizes=decoder_hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            negative_slope=float(negative_slope),
        ).float()
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _array(data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor, tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    def _run_epoch(self, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module) -> float:
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

    def encode(self, data, batch_size: int = 32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        encoded = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                encoded.append(self.model.encoder(xb.float()).detach().numpy())
        return np.concatenate(encoded, axis=0)

    def encode_decode(self, data, batch_size: int = 32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        decoded = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                decoded.append(self.model(xb.float()).detach().numpy())
        return np.concatenate(decoded, axis=0)


def autoenc_create(
    input_size,
    encoding_size,
    encoder_hidden_sizes=None,
    decoder_hidden_sizes=None,
    activation="relu",
    output_activation="none",
    negative_slope=0.2,
    validation_strategy="static",
    stopping_rule="none",
):
    return DenseAutoencoderModel(
        input_size,
        encoding_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        activation=activation,
        output_activation=output_activation,
        negative_slope=negative_slope,
        validation_strategy=validation_strategy,
        stopping_rule=stopping_rule,
    )


def autoenc_fit(
    autoencoder,
    data,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    validation_strategy="static",
    stopping_rule="none",
    val_ratio=0.3,
    patience=100,
    min_delta=1e-4,
    sma_window=5,
    ema_alpha=0.2,
    test_window=30,
    p_value=0.05,
):
    autoencoder.validation_strategy, autoencoder.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=autoencoder.validation_strategy,
        stopping_rule=autoencoder.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
    )
    autoencoder.fit(data, config)
    return autoencoder, np.array(autoencoder.train_loss), np.array(autoencoder.val_loss)


def autoenc_encode(autoencoder, data, batch_size=32):
    return autoencoder.encode(data, batch_size=batch_size)


def autoenc_encode_decode(autoencoder, data, batch_size=32):
    return autoencoder.encode_decode(data, batch_size=batch_size)
