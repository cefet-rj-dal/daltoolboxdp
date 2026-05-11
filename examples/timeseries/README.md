# Time Series — Examples

These examples show how the Python-backed models in `daltoolboxdp` fit into a time-series workflow based on sliding windows. They are the shortest path if the goal is to understand the end-to-end transformation from a raw series into latent representations and reconstructed windows.

Use this section when you want to see:

- how a univariate series is converted into windows of length `p`
- how an autoencoder compresses those windows into `k` latent dimensions
- how reconstruction error helps evaluate the quality of the learned representation

New forecasting flexibility
- `torch_ts_mlp` now exposes hidden activation, output activation, normalization (`"none"`, `"batch"`, `"layer"`), and initialization strategy.
- `ts_lstm` now supports configurable `hidden_size`, `sequence_length`, recurrent depth, dropout, bidirectionality, and an optional dense head.
- `ts_conv1d` now supports explicit channel/sequence reshaping, multiple convolutional blocks, optional pooling, and a configurable dense head.

- [ts_encode.md](ts_encode.md) — Encodes windows (p->k) using an autoencoder; the bottleneck provides a compact representation for downstream use.
- [ts_encode-decode.md](ts_encode-decode.md) — Encodes and reconstructs (p<->k) to evaluate quality via reconstruction error.
- [ts_lstm.md](ts_lstm.md) — Forecasts one step ahead from lagged windows using a PyTorch LSTM with unified validation/stopping options.
- [ts_conv1d.md](ts_conv1d.md) — Forecasts one step ahead from lagged windows using a PyTorch Conv1D model.
- [torch_ts_mlp.md](torch_ts_mlp.md) — Forecasts one step ahead from lagged windows using a feedforward PyTorch MLP.
