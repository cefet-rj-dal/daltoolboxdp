## Feature Selection with Lasso

This example uses Lasso (L1 regularization) to select relevant features based on their relationship to the target variable. L1 induces sparsity in coefficients, removing less important features.

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
# Example data
iris <- datasets::iris
```


``` r
# Lasso - step by step

# 1) Fit the selector with target "Species"
myfeature <- fit(fs_lasso("Species"), iris)

# 2) View selected features
print(myfeature$features)
```

```
## [1] "Sepal.Width"  "Petal.Length" "Petal.Width"
```

``` r
# 3) Transform data to keep selected features + target
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Sepal.Width Petal.Length Petal.Width Species
## 1         3.5          1.4         0.2  setosa
## 2         3.0          1.4         0.2  setosa
## 3         3.2          1.3         0.2  setosa
## 4         3.1          1.5         0.2  setosa
## 5         3.6          1.4         0.2  setosa
## 6         3.9          1.7         0.4  setosa
```

