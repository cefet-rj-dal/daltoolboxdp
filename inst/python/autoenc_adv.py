"""
Unified adversarial autoencoder used by daltoolboxdp via reticulate.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from autoenc_common import AutoencTrainingConfig, StopController, split_indices, validate_strategy


class QNet(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.lin1 = nn.Linear(int(input_size), 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, int(encoding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.dropout(self.lin1(x), p=0.4, training=self.training))
        x = F.relu(F.dropout(self.lin2(x), p=0.4, training=self.training))
        return self.lin3(x)


class PNet(nn.Module):
    def __init__(self, input_size: int, encoding_size: int):
        super().__init__()
        self.lin1 = nn.Linear(int(encoding_size), 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, int(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.dropout(self.lin1(x), p=0.4, training=self.training))
        x = F.dropout(self.lin2(x), p=0.4, training=self.training)
        return torch.sigmoid(self.lin3(x))


class DNetGauss(nn.Module):
    def __init__(self, encoding_size: int):
        super().__init__()
        self.lin1 = nn.Linear(int(encoding_size), 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.dropout(self.lin1(x), p=0.4, training=self.training))
        x = F.relu(F.dropout(self.lin2(x), p=0.4, training=self.training))
        return torch.sigmoid(self.lin3(x))


class AdversarialAutoencoderModel:
    def __init__(self, input_size: int, encoding_size: int, validation_strategy: str = "static", stopping_rule: str = "none"):
        self.validation_strategy, self.stopping_rule = validate_strategy(validation_strategy, stopping_rule)
        self.input_size = int(input_size)
        self.encoding_size = int(encoding_size)
        self.Q = QNet(input_size, encoding_size)
        self.P = PNet(input_size, encoding_size)
        self.D_gauss = DNetGauss(encoding_size)
        self.encoder_opt = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
        self.decoder_opt = torch.optim.Adam(self.P.parameters(), lr=0.0001)
        self.generator_opt = torch.optim.Adam(self.Q.parameters(), lr=0.00005)
        self.discriminator_opt = torch.optim.Adam(self.D_gauss.parameters(), lr=0.00005)
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs_done: int = 0

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

    def _train_epoch(self, loader):
        tiny = 1e-15
        losses = []
        for xb, _ in loader:
            xb = xb.float().view(xb.size(0), -1)

            z_sample = self.Q(xb)
            x_sample = self.P(z_sample)
            recon_loss = nn.MSELoss()(x_sample + tiny, xb + tiny)
            recon_loss.backward()
            self.decoder_opt.step()
            self.encoder_opt.step()
            self.P.zero_grad(); self.Q.zero_grad(); self.D_gauss.zero_grad()

            self.Q.eval()
            z_real = torch.randn(len(xb), self.encoding_size) * 5.0
            z_fake = self.Q(xb)
            d_real = self.D_gauss(z_real)
            d_fake = self.D_gauss(z_fake)
            d_loss = -torch.mean(torch.log(d_real + tiny) + torch.log(1 - d_fake + tiny))
            d_loss.backward()
            self.discriminator_opt.step()
            self.P.zero_grad(); self.Q.zero_grad(); self.D_gauss.zero_grad()

            self.Q.train()
            z_fake = self.Q(xb)
            d_fake = self.D_gauss(z_fake)
            g_loss = -torch.mean(torch.log(d_fake + tiny))
            g_loss.backward()
            self.generator_opt.step()
            self.P.zero_grad(); self.Q.zero_grad(); self.D_gauss.zero_grad()
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

    def _load_state(self, state):
        self.Q.load_state_dict(state["Q"])
        self.P.load_state_dict(state["P"])
        self.D_gauss.load_state_dict(state["D"])

    def fit(self, data, config: AutoencTrainingConfig):
        if config.seed is not None:
            np.random.seed(int(config.seed))
            torch.manual_seed(int(config.seed))
        array = self._array(data)
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
            self.train_loss.append(self._train_epoch(train_loader))
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                self.val_loss.append(val_loss)
                if self.stopping_rule == "none":
                    stopper.best_state = self._state_dict()
                else:
                    current = val_loss
                    stopper.val_history.append(float(current))
                    if stopper.rule == "h":
                        improved = stopper._h_improved()
                        if improved:
                            stopper.best_state = self._state_dict()
                            stopper.patience_ctr = 0
                        else:
                            stopper.patience_ctr += 1
                        should_stop = stopper.patience_ctr >= stopper.patience
                    else:
                        if stopper.rule == "patience":
                            monitor = current
                        elif stopper.rule == "sma":
                            monitor = float(np.mean(stopper.val_history[-stopper.sma_window :]))
                        else:
                            if stopper.ema_value is None:
                                stopper.ema_value = current
                            else:
                                stopper.ema_value = stopper.ema_alpha * current + (1.0 - stopper.ema_alpha) * stopper.ema_value
                            monitor = float(stopper.ema_value)
                        if (stopper.best_value - monitor) > stopper.min_delta:
                            stopper.best_value = monitor
                            stopper.best_state = self._state_dict()
                            stopper.patience_ctr = 0
                        else:
                            stopper.patience_ctr += 1
                        should_stop = stopper.patience_ctr >= stopper.patience
                    if should_stop:
                        break
        if stopper.best_state is not None:
            self._load_state(stopper.best_state)
        return self

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


def autoenc_adv_create(input_size, encoding_size, validation_strategy="static", stopping_rule="none"):
    return AdversarialAutoencoderModel(input_size, encoding_size, validation_strategy=validation_strategy, stopping_rule=stopping_rule)


def autoenc_adv_fit(aae, data, batch_size=350, num_epochs=100, learning_rate=0.001, validation_strategy="static", stopping_rule="none", val_ratio=0.3, patience=100, min_delta=1e-4, sma_window=5, ema_alpha=0.2, test_window=30, p_value=0.05, seed=42):
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
        seed=None if seed is None else int(seed),
    )
    aae.fit(data, config)
    return aae, np.array(aae.train_loss), np.array(aae.val_loss)


def autoenc_adv_encode(aae, data, batch_size=32):
    return aae.encode(data, batch_size=batch_size)


def autoenc_adv_encode_decode(aae, data, batch_size=350):
    return aae.encode_decode(data, batch_size=batch_size)
