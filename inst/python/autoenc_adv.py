"""
Unified adversarial autoencoder used by daltoolboxdp via reticulate.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, ensure_int_list, split_indices, validate_strategy


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    name = str(name).lower()
    if name == "relu":
        return F.relu(x)
    if name == "leaky_relu":
        return F.leaky_relu(x, negative_slope=0.2)
    if name == "elu":
        return F.elu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name}")


class FeedForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes, output_size: int, activation: str = "relu", dropout: float = 0.4, output_activation: str = "none"):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = int(input_size)
        for hidden_size in ensure_int_list(hidden_sizes, allow_empty=True):
            self.layers.append(nn.Linear(prev, int(hidden_size)))
            prev = int(hidden_size)
        self.output = nn.Linear(prev, int(output_size))
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.output_activation = str(output_activation).lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = _activation(self.activation, layer(x))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x


class AdversarialAutoencoderModel:
    def __init__(
        self,
        input_size: int,
        encoding_size: int,
        encoder_hidden_sizes=None,
        decoder_hidden_sizes=None,
        discriminator_hidden_sizes=None,
        activation: str = "relu",
        dropout: float = 0.4,
        latent_prior_scale: float = 5.0,
        lr_encoder: Optional[float] = None,
        lr_decoder: Optional[float] = None,
        lr_generator: Optional[float] = None,
        lr_discriminator: Optional[float] = None,
        validation_strategy: str = "static",
        stopping_rule: str = "none",
    ):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.encoder_hidden_sizes = [60, 60] if encoder_hidden_sizes is None else encoder_hidden_sizes
        self.decoder_hidden_sizes = [60, 60] if decoder_hidden_sizes is None else decoder_hidden_sizes
        self.discriminator_hidden_sizes = [60, 60] if discriminator_hidden_sizes is None else discriminator_hidden_sizes
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.latent_prior_scale = float(latent_prior_scale)
        self.lr_encoder = None if lr_encoder is None else float(lr_encoder)
        self.lr_decoder = None if lr_decoder is None else float(lr_decoder)
        self.lr_generator = None if lr_generator is None else float(lr_generator)
        self.lr_discriminator = None if lr_discriminator is None else float(lr_discriminator)

        self.Q = FeedForwardNet(self.input_size, self.encoder_hidden_sizes, self.encoding_size, activation=self.activation, dropout=self.dropout)
        self.P = FeedForwardNet(self.encoding_size, self.decoder_hidden_sizes, self.input_size, activation=self.activation, dropout=self.dropout, output_activation="sigmoid")
        self.D_gauss = FeedForwardNet(self.encoding_size, self.discriminator_hidden_sizes, 1, activation=self.activation, dropout=self.dropout, output_activation="sigmoid")
        self._reset_optimizers()

        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

    def _reset_optimizers(self):
        encoder_lr = 1e-4 if self.lr_encoder is None else self.lr_encoder
        decoder_lr = 1e-4 if self.lr_decoder is None else self.lr_decoder
        generator_lr = 5e-5 if self.lr_generator is None else self.lr_generator
        discriminator_lr = 5e-5 if self.lr_discriminator is None else self.lr_discriminator
        self.encoder_opt = torch.optim.Adam(self.Q.parameters(), lr=encoder_lr)
        self.decoder_opt = torch.optim.Adam(self.P.parameters(), lr=decoder_lr)
        self.generator_opt = torch.optim.Adam(self.Q.parameters(), lr=generator_lr)
        self.discriminator_opt = torch.optim.Adam(self.D_gauss.parameters(), lr=discriminator_lr)

    @staticmethod
    def _array(data):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _loader(array: np.ndarray, batch_size: int, shuffle: bool):
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor, tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)

    def _reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.Q(x)
        recon = self.P(z)
        return nn.MSELoss()(recon, x)

    def _zero_all(self):
        self.P.zero_grad()
        self.Q.zero_grad()
        self.D_gauss.zero_grad()

    def _train_epoch(self, loader):
        tiny = 1e-15
        losses = []
        self.Q.train()
        self.P.train()
        self.D_gauss.train()
        for xb, _ in loader:
            xb = xb.float().view(xb.size(0), -1)

            z_sample = self.Q(xb)
            x_sample = self.P(z_sample)
            recon_loss = nn.MSELoss()(x_sample + tiny, xb + tiny)
            recon_loss.backward()
            self.decoder_opt.step()
            self.encoder_opt.step()
            self._zero_all()

            self.Q.eval()
            z_real = torch.randn(len(xb), self.encoding_size) * self.latent_prior_scale
            z_fake = self.Q(xb)
            d_real = self.D_gauss(z_real)
            d_fake = self.D_gauss(z_fake)
            d_loss = -torch.mean(torch.log(d_real + tiny) + torch.log(1 - d_fake + tiny))
            d_loss.backward()
            self.discriminator_opt.step()
            self._zero_all()

            self.Q.train()
            z_fake = self.Q(xb)
            d_fake = self.D_gauss(z_fake)
            g_loss = -torch.mean(torch.log(d_fake + tiny))
            g_loss.backward()
            self.generator_opt.step()
            self._zero_all()
            losses.append(float(recon_loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _eval_epoch(self, loader):
        losses = []
        self.Q.eval()
        self.P.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.float().view(xb.size(0), -1)
                z = self.Q(xb)
                recon = self.P(z)
                losses.append(float(nn.MSELoss()(recon, xb).item()))
        return float(np.mean(losses)) if losses else 0.0

    def _state_dict(self):
        return {
            "Q": {k: v.detach().clone() for k, v in self.Q.state_dict().items()},
            "P": {k: v.detach().clone() for k, v in self.P.state_dict().items()},
            "D": {k: v.detach().clone() for k, v in self.D_gauss.state_dict().items()},
        }

    def _load_state(self, state: Dict[str, Dict[str, torch.Tensor]]):
        self.Q.load_state_dict(state["Q"])
        self.P.load_state_dict(state["P"])
        self.D_gauss.load_state_dict(state["D"])

    def fit(self, data, config: AutoencTrainingConfig):

        if float(config.learning_rate) > 0:
            base_lr = float(config.learning_rate)
            if self.lr_encoder is None:
                self.lr_encoder = base_lr
            if self.lr_decoder is None:
                self.lr_decoder = base_lr
            if self.lr_generator is None:
                self.lr_generator = base_lr * 0.5
            if self.lr_discriminator is None:
                self.lr_discriminator = base_lr * 0.5
        self._reset_optimizers()

        array = self._array(data)
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
            self.train_loss.append(self._train_epoch(train_loader))
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                self.val_loss.append(val_loss)
                if self.stopping_rule == "none":
                    stopper.best_state = self._state_dict()
                elif stopper.step(self, val_loss):
                    break
        if stopper.best_state is not None:
            self._load_state(stopper.best_state)
        return self

    def state_dict(self):
        # StopController clones model.state_dict() by default. For AAE the model spans three modules.
        return self._state_dict()

    def load_state_dict(self, state):
        self._load_state(state)

    def encode(self, data, batch_size=32):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.Q.eval()
        with torch.no_grad():
            for xb, _ in loader:
                outs.append(self.Q(xb.float().view(xb.size(0), -1)).detach().numpy())
        return np.concatenate(outs, axis=0)

    def encode_decode(self, data, batch_size=350):
        array = self._array(data)
        loader = self._loader(array, batch_size, False)
        outs = []
        self.Q.eval()
        self.P.eval()
        with torch.no_grad():
            for xb, _ in loader:
                flat = xb.float().view(xb.size(0), -1)
                outs.append(self.P(self.Q(flat)).detach().numpy())
        return np.concatenate(outs, axis=0)


def autoenc_adv_create(
    input_size,
    encoding_size,
    encoder_hidden_sizes=None,
    decoder_hidden_sizes=None,
    discriminator_hidden_sizes=None,
    activation="relu",
    dropout=0.4,
    latent_prior_scale=5.0,
    lr_encoder=None,
    lr_decoder=None,
    lr_generator=None,
    lr_discriminator=None,
    validation_strategy="static",
    stopping_rule="none",
):
    return AdversarialAutoencoderModel(
        input_size,
        encoding_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=decoder_hidden_sizes,
        discriminator_hidden_sizes=discriminator_hidden_sizes,
        activation=activation,
        dropout=dropout,
        latent_prior_scale=latent_prior_scale,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        lr_generator=lr_generator,
        lr_discriminator=lr_discriminator,
        validation_strategy=validation_strategy,
        stopping_rule=stopping_rule,
    )


def autoenc_adv_fit(aae, data, batch_size=350, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05):
    aae.validation_strategy, aae.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
    config = AutoencTrainingConfig(
        batch_size=int(batch_size),
        num_epochs=int(num_epochs),
        learning_rate=float(learning_rate),
        validation_strategy=aae.validation_strategy,
        stopping_rule=aae.stopping_rule,
        val_ratio=float(val_ratio),
        patience=int(patience),
        min_delta=float(min_delta),
        sma_window=int(sma_window),
        ema_alpha=float(ema_alpha),
        test_window=int(test_window),
        p_value=float(p_value),
    )
    aae.fit(data, config)
    return aae, np.array(aae.train_loss), np.array(aae.val_loss)


def autoenc_adv_encode(aae, data, batch_size=32):
    return aae.encode(data, batch_size=batch_size)


def autoenc_adv_encode_decode(aae, data, batch_size=350):
    return aae.encode_decode(data, batch_size=batch_size)
