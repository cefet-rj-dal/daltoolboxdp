"""
Unified stacked autoencoder used by daltoolboxdp via reticulate.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, build_dense_stack, ensure_int_list, split_indices, validate_strategy


class StackUnit(nn.Module):
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


class StackedAutoencoderModel:
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        k: int = 3,
        encoding_sizes=None,
        encoder_hidden_sizes=None,
        decoder_hidden_sizes=None,
        activation: str = "relu",
        output_activation: str = "none",
        negative_slope: float = 0.2,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
    ):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.encoding_sizes = ensure_int_list(encoding_sizes, default=[int(encoding_size)] * int(k))
        self.stage_count = len(self.encoding_sizes)
        self.stack: List[StackUnit] = []
        current_size = int(input_size)
        encoder_hidden_per_stage = self._expand_stage_param(encoder_hidden_sizes, self.stage_count, default=[64])
        decoder_hidden_per_stage = self._expand_stage_param(decoder_hidden_sizes, self.stage_count, default=None)
        for idx, latent_size in enumerate(self.encoding_sizes):
            unit = StackUnit(
                current_size,
                latent_size,
                encoder_hidden_sizes=encoder_hidden_per_stage[idx],
                decoder_hidden_sizes=decoder_hidden_per_stage[idx],
                activation=activation,
                output_activation=output_activation,
                negative_slope=negative_slope,
            ).float()
            self.stack.append(unit)
            current_size = int(latent_size)
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    @staticmethod
    def _expand_stage_param(value, n_stages: int, default=None):
        if value is None:
            return [default for _ in range(n_stages)]
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple, np.ndarray)):
            if len(value) != n_stages:
                raise ValueError("Stage-specific parameter lists must match the number of stacked stages.")
            return [list(v) if v is not None else default for v in value]
        return [value for _ in range(n_stages)]

    @staticmethod
    def _array(data):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _loader(array: np.ndarray, batch_size: int, shuffle: bool):
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor, tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    @staticmethod
    def _run_epoch(unit: nn.Module, loader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module):
        losses: List[float] = []
        if optimizer is None:
            unit.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    losses.append(float(criterion(unit(xb.float()), yb.float()).item()))
        else:
            unit.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(unit(xb.float()), yb.float())
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _fit_unit(self, unit: nn.Module, array: np.ndarray, config: AutoencTrainingConfig):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(unit.parameters(), lr=float(config.learning_rate))
        stopper = StopController(self.stopping_rule, config.min_delta, config.patience, config.sma_window, config.ema_alpha, config.test_window, config.p_value)
        train_hist: List[float] = []
        val_hist: List[float] = []
        epochs_done = 0

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
            epochs_done += 1
            if self.validation_strategy == "dynamic":
                train_idx, val_idx = split_indices(array.shape[0], config.val_ratio)
                train_loader = self._loader(array[train_idx], config.batch_size, True)
                val_loader = self._loader(array[val_idx], config.batch_size, False)
            train_hist.append(self._run_epoch(unit, train_loader, optimizer, criterion))
            if val_loader is not None:
                val_loss = self._run_epoch(unit, val_loader, None, criterion)
                val_hist.append(val_loss)
                if stopper.step(unit, val_loss):
                    break
        if stopper.best_state is not None:
            unit.load_state_dict(stopper.best_state)
        return train_hist, val_hist, epochs_done

    def _encode_decode_unit(self, unit: nn.Module, array: np.ndarray, batch_size: int):
        loader = self._loader(array, batch_size, False)
        outs = []
        unit.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(unit(xb.float()).detach().numpy())
        return np.concatenate(outs, axis=0)

    def _encode_unit(self, unit: nn.Module, array: np.ndarray, batch_size: int):
        loader = self._loader(array, batch_size, False)
        outs = []
        unit.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(unit.encoder(xb.float()).detach().numpy())
        return np.concatenate(outs, axis=0)

    def _decode_unit(self, unit: nn.Module, array: np.ndarray, batch_size: int):
        loader = self._loader(array, batch_size, False)
        outs = []
        unit.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(unit.decoder(xb.float()).detach().numpy())
        return np.concatenate(outs, axis=0)

    def fit(self, data, config: AutoencTrainingConfig):
        current = self._array(data)
        self.train_loss = []
        self.val_loss = []
        self.epochs_done = 0
        for unit in self.stack:
            train_hist, val_hist, done = self._fit_unit(unit, current, config)
            self.train_loss = train_hist
            self.val_loss = val_hist
            self.epochs_done += done
            current = self._encode_unit(unit, current, config.batch_size)
        return self

    def encode(self, data, batch_size=32):
        array = self._array(data)
        current = array
        for unit in self.stack:
            current = self._encode_unit(unit, current, batch_size)
        return current

    def encode_decode(self, data, batch_size=32):
        array = self._array(data)
        encoded = array
        for unit in self.stack:
            encoded = self._encode_unit(unit, encoded, batch_size)
        reconstructed = encoded
        for unit in reversed(self.stack):
            reconstructed = self._decode_unit(unit, reconstructed, batch_size)
        return reconstructed


def autoenc_stacked_create(
    input_size,
    encoding_size,
    k=3,
    encoding_sizes=None,
    encoder_hidden_sizes=None,
    decoder_hidden_sizes=None,
    activation="relu",
    output_activation="none",
    negative_slope=0.2,
    validation_strategy="static",
    stopping_rule="none",
):
    return StackedAutoencoderModel(
        input_size,
        encoding_size,
        k=k,
        encoding_sizes=encoding_sizes,
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        activation=activation,
        output_activation=output_activation,
        negative_slope=negative_slope,
        validation_strategy=validation_strategy,
        stopping_rule=stopping_rule,
    )


def autoenc_stacked_fit(stack, data, batch_size=32, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05):
    stack.validation_strategy, stack.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=stack.validation_strategy,
        stopping_rule=stack.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
    )
    stack.fit(data, config)
    return stack, np.array(stack.train_loss), np.array(stack.val_loss)


def autoenc_stacked_encode(sae, data, batch_size=32):
    return sae.encode(data, batch_size=batch_size)


def autoenc_stacked_encode_decode(sae, data, batch_size=32):
    return sae.encode_decode(data, batch_size=batch_size)
