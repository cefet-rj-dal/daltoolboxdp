# DAL Toolbox DP Examples

This directory contains the source `Rmd` files used to generate the example pages under `examples/`. If you are exploring the package for the first time, the best path is to choose a topic below and open the generated example index in `examples/`, where the rendered `.md` files live.

The topics are organized to answer different questions:

- Which Python-backed autoencoder should I use to compress time-series windows?
- Which scikit-learn classifier wrappers are available in the `daltoolbox` architecture?
- How do the time-series examples use these components for encoding and reconstruction?

- [autoencoder](examples/autoencoder/README.md) — Autoencoders for time-series windows: simple, convolutional, denoising, LSTM, stacked, and variational variants, in both encode and encode-decode forms.
- [classification](examples/classification/README.md) — Classification wrappers backed by Python libraries, covering model creation, training, prediction, and evaluation.
- [timeseries](examples/timeseries/README.md) — Time-series examples focused on encoding (p->k) and reconstruction (k->p) with autoencoders.
