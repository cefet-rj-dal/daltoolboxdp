# DAL Toolbox DP Examples

This directory contains the source `Rmd` files used to generate the example pages under `examples/`. If you are exploring the package for the first time, the best path is to choose a topic below and open the generated example index in `examples/`, where the rendered `.md` files live.

The topics are organized to answer different questions:

- Which Python-backed autoencoder should I use for representation only versus reconstruction-aware workflows?
- Which classification examples are grouped under scikit-learn and PyTorch neural models, now using the same class-score output contract as `daltoolbox`?
- Which PyTorch regression examples are available for base, static-validation, and dynamic-validation workflows?
- How do the time-series examples separate representation/preprocessing, direct prediction, and LSTM training regimes?

- [autoencoder](autoencoder/README.md) — Autoencoder examples split between encoder-only use cases and encoder-decoder reconstruction workflows.
- [classification](classification/README.md) — Classification examples grouped into scikit-learn (`01`-`06`) and PyTorch (`07`-`08`) blocks.
- [regression](regression/README.md) — Regression examples organized around the PyTorch MLP regressor, including base, static-validation, and dynamic-validation variants.
- [timeseries](timeseries/README.md) — Time-series examples grouped into representation/preprocessing, direct prediction, and LSTM early-stopping regimes.
