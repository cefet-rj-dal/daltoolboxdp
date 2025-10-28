## Class Balance: Subsampling

This example shows how to handle class imbalance by applying subsampling (reduce the majority class) on an imbalanced subset of the Iris dataset.

Prerequisites
- R packages: daltoolbox, daltoolboxdp


``` r
# Installation (if needed)
#install.packages("daltoolboxdp")
```


``` r
# Loading packages
library(daltoolbox)
library(daltoolboxdp)
```


``` r
# Example data and creation of artificial imbalance
iris <- datasets::iris
data(iris)
mod_iris <- iris[c(1:50,51:71,101:111),]
table(mod_iris$Species)                      # original distribution
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```


``` r
# Subsampling - reduce the majority class to balance
bal <- bal_subsampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)                    # distribution after subsampling
```

```
## 
##     setosa versicolor  virginica 
##         11         11         11
```

