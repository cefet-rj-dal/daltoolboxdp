# Time Series — Examples

These examples show how the Python-backed models in `daltoolboxdp` fit into a time-series workflow based on sliding windows. They are organized in three groups: representation/preprocessing, direct prediction, and training-regime studies for the LSTM forecaster.

Use this section when you want to see:

- how a univariate series is converted into windows of length `p`
- how latent representations and reconstructions work before forecasting
- how direct forecasting differs across MLP, Conv1D, and LSTM models
- how static and dynamic validation affect LSTM training

New forecasting flexibility
- `torch_ts_mlp` now exposes hidden activation, output activation, normalization (`"none"`, `"batch"`, `"layer"`), and initialization strategy.
- `ts_lstm` now supports configurable `hidden_size`, `sequence_length`, recurrent depth, dropout, bidirectionality, and an optional dense head.
- `ts_conv1d` now supports explicit channel/sequence reshaping, multiple convolutional blocks, optional pooling, and a configurable dense head.

Representation / preprocessing
- [01_ts_encode.md](01_ts_encode.md) — Encodes windows (p->k) into compact latent vectors for downstream use.
- [02_ts_encode-decode.md](02_ts_encode-decode.md) — Encodes and reconstructs (p<->k) to inspect reconstruction quality before forecasting tasks.

Prediction
- [11_torch_ts_mlp.md](11_torch_ts_mlp.md) — One-step-ahead forecasting with a feedforward PyTorch MLP.
- [12_ts_conv1d.md](12_ts_conv1d.md) — One-step-ahead forecasting with a PyTorch Conv1D model.
- [13_ts_lstm.md](13_ts_lstm.md) — One-step-ahead forecasting with a PyTorch LSTM under the standard training flow.

LSTM training regimes
- [21_ts_lstm_static_patience.md](21_ts_lstm_static_patience.md) — LSTM forecasting with static validation and patience-based early stopping.
- [22_ts_lstm_dynamic_patience.md](22_ts_lstm_dynamic_patience.md) — LSTM forecasting with dynamic validation and patience-based early stopping.
