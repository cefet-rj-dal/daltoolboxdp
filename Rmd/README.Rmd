# DAL Toolbox DP Examples

This directory contains the source `Rmd` files used to generate the example pages under `examples/`. If you are exploring the package for the first time, the best path is to choose a topic below and open the generated example index in `examples/`, where the rendered `.md` files live.

The topics are organized to answer different questions:

- Which Python-backed autoencoder should I use to compress time-series windows?
- Which scikit-learn classifier wrappers are available in the `daltoolbox` architecture?
- Which Python-backed regression wrappers are available for numeric prediction?
- How do the time-series examples cover both representation learning and direct forecasting?

- [autoencoder](autoencoder/README.md) — Autoencoders for time-series windows: simple, convolutional, denoising, LSTM, stacked, and variational variants, in both encode and encode-decode forms.
- [classification](classification/README.md) — Classification wrappers backed by Python libraries, including scikit-learn and PyTorch neural models.
- [regression](regression/README.md) — Regression wrappers backed by Python libraries, currently including the PyTorch MLP regressor.
- [timeseries](timeseries/README.md) — Time-series examples for encoding, reconstruction, and direct forecasting with PyTorch models.
