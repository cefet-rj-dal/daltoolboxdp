# Time Series — Examples

These examples show how the Python-backed models in `daltoolboxdp` fit into a time-series workflow based on sliding windows. They are the shortest path if the goal is to understand the end-to-end transformation from a raw series into latent representations and reconstructed windows.

Use this section when you want to see:

- how a univariate series is converted into windows of length `p`
- how an autoencoder compresses those windows into `k` latent dimensions
- how reconstruction error helps evaluate the quality of the learned representation

- [ts_encode.md](ts_encode.md) — Encodes windows (p->k) using an autoencoder; the bottleneck provides a compact representation for downstream use.
- [ts_encode-decode.md](ts_encode-decode.md) — Encodes and reconstructs (p<->k) to evaluate quality via reconstruction error.
