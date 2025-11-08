
<!-- README.md is generated from README.Rmd. Please edit that file -->

# <img src='https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/master/inst/logo.png' alt='Logo do pacote daltoolboxdp' align='centre' height='150' width='139'/> Deep Python Extensions for Data Analytics Workflows

<!-- badges: start -->

![GitHub
Stars](https://img.shields.io/github/stars/cefet-rj-dal/daltoolboxdp?logo=Github)
![CRAN Downloads](https://cranlogs.r-pkg.org/badges/daltoolboxdp)
<!-- badges: end -->

daltoolboxdp provides Python extensions to enhance data analytics workflows, 
especially deep learning. It includes tools for:

- Data sampling and transformation  
- Feature selection 
- Machine learning
- Deep learning
- Balancing strategies (e.g., SMOTE)  
- Model construction and tuning

These capabilities leverage Python libraries via the `reticulate`
interface, enabling seamless integration with the broader Python machine
learning ecosystem. The package supports instance selection and hybrid
workflows that combine R and Python functionalities for flexible and
reproducible analytical pipelines.

The architecture is inspired by the **Experiment Lines** approach, which
promotes modularity, extensibility, and interoperability across tools.  
More information on Experiment Lines is available in [Ogasawara et
al. (2009)](https://doi.org/10.1007/978-3-642-02279-1_20).

------------------------------------------------------------------------

# Examples

Example scripts are available at:

- [Classification](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/classification)
- [Autoenconders](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/autoencoder)
- [Feature
  Selection](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/feature-selection)
- [Time
  Series](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/timeseries)
- [Sampling](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples/sampling)

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
