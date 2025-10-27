## Feature Selection with Relief

This example uses the Relief method to estimate feature relevance by considering nearest neighbors and differences across classes, ranking and selecting the most informative features for the target.

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
# Relief - step by step

# 1) Fit the selector with target "Species"
myfeature <- fit(fs_relief("Species"), iris)

# 2) View selected features
print(myfeature$features)
```

```
## [1] "Petal.Length" "Petal.Width"
```

``` r
# 3) Transform data to keep selected features + target
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Petal.Length Petal.Width Species
## 1          1.4         0.2  setosa
## 2          1.4         0.2  setosa
## 3          1.3         0.2  setosa
## 4          1.5         0.2  setosa
## 5          1.4         0.2  setosa
## 6          1.7         0.4  setosa
```

