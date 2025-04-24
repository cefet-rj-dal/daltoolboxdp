
<!-- README.md is generated from README.Rmd. Please edit that file -->

# <img src='https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/master/inst/logo.png' alt='Logo do pacote daltoolboxdp' align='centre' height='150' width='139'/> daltoolboxdp

<!-- badges: start -->

![GitHub
Stars](https://img.shields.io/github/stars/cefet-rj-dal/daltoolboxdp?logo=Github)
![CRAN Downloads](https://cranlogs.r-pkg.org/badges/daltoolboxdp)
<!-- badges: end -->

**daltoolboxdp** provides Python-based extensions to enhance data
analytics workflows, particularly for tasks involving data preprocessing
and predictive modeling. It includes tools for:

- Data sampling and transformation  
- Feature selection  
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

## Installation

You can install the latest stable version from CRAN:

``` r
install.packages("daltoolboxdp")
```

To install the development version from GitHub:

``` r
# install.packages("devtools")
library(devtools)
devtools::install_github("cefet-rj-dal/daltoolboxdp", force = TRUE, dependencies = FALSE, upgrade = "never")
```

------------------------------------------------------------------------

## Examples

Example scripts are available at:

- [daltoolboxdp/examples](https://github.com/cefet-rj-dal/daltoolboxdp/tree/main/examples)

``` r
library(daltoolboxdp)
#> Registered S3 method overwritten by 'quantmod':
#>   method            from
#>   as.zoo.data.frame zoo
#> Registered S3 methods overwritten by 'forecast':
#>   method  from 
#>   head.ts stats
#>   tail.ts stats

# Example usage (replace with actual function when available)
# e.g., data <- my_sampler_function(data, method = "undersample")
```

------------------------------------------------------------------------

## Bug reports and feature requests

Please report issues or suggest new features via:

- [GitHub Issues](https://github.com/cefet-rj-dal/daltoolboxdp/issues)
