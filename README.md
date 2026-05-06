
<!-- README.md is generated from README.Rmd. Please edit that file -->

# <img src='https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/master/inst/logo.png' alt='Logo do pacote daltoolboxdp' align='centre' height='150' width='139'/> DAL Toolbox Deep Python

<!-- badges: start -->

![GitHub
Stars](https://img.shields.io/github/stars/cefet-rj-dal/daltoolboxdp?logo=Github)
![CRAN Downloads](https://cranlogs.r-pkg.org/badges/daltoolboxdp)
<!-- badges: end -->

daltoolboxdp extends `daltoolbox` with Python-backed components, with
emphasis on deep learning and Python-native modeling. In the package
name, `dp` stands for **Deep Python**.

It currently focuses on:

- Deep learning models backed by `torch`
- Scikit-learn classifiers exposed through the `daltoolbox` API
- Time-series forecasting models backed by Python
- Integration of Python model objects into the `daltoolbox` architecture

These capabilities rely on the `reticulate` bridge, so the package can
keep the object and workflow conventions of `daltoolbox` while
delegating training, encoding, and prediction to Python libraries such
as `torch` and `scikit-learn`.

The architecture is inspired by the **Experiment Lines** approach, which
promotes modularity, extensibility, and interoperability across tools.\
More information on Experiment Lines is available in [Ogasawara et
al. (2009)](https://doi.org/10.1007/978-3-642-02279-1_20).

------------------------------------------------------------------------

# Examples

The example set is organized by topic and generated from the source
files under `Rmd/`. If you are exploring the package for the first time,
start from the rendered indexes under `examples/`.

The current topics are organized around these questions:

- Which Python-backed autoencoder should I use to compress time-series
  windows?
- Which scikit-learn classifier wrappers are available in the
  `daltoolbox` architecture?
- Which Python-backed regression wrappers are available for numeric
  prediction?
- How do the time-series examples cover both representation learning and
  direct forecasting?

Rendered examples are available at:

- [Autoencoders](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/autoencoder) -
  Autoencoders for time-series windows: simple, convolutional,
  denoising, LSTM, stacked, and variational variants, in both encode and
  encode-decode forms.
- [Classification](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/classification) -
  Classification wrappers backed by Python libraries, including
  scikit-learn and PyTorch neural models.
- [Regression](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/regression) -
  Regression wrappers backed by Python libraries, currently including
  the PyTorch MLP regressor.
- [Time
  Series](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/timeseries) -
  Time-series examples for encoding, reconstruction, and direct
  forecasting with PyTorch models.

------------------------------------------------------------------------

# Installation

You can install the latest stable version from CRAN:

``` r
install.packages("daltoolboxdp")
```

To install the development version from GitHub:

``` r
library(devtools)
devtools::install_github("cefet-rj-dal/daltoolboxdp", force = TRUE, dependencies = FALSE, upgrade = "never")
```

------------------------------------------------------------------------

# Bug reports and feature requests

Please report issues or suggest new features via:

- [GitHub Issues](https://github.com/cefet-rj-dal/daltoolboxdp/issues)
