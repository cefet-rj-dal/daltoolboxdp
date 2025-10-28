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
## [1] "Petal.Width"  "Petal.Length"
```

``` r
# 3) Transform data to keep selected features + target
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Petal.Width Petal.Length Species
## 1         0.2          1.4  setosa
## 2         0.2          1.4  setosa
## 3         0.2          1.3  setosa
## 4         0.2          1.5  setosa
## 5         0.2          1.4  setosa
## 6         0.4          1.7  setosa
```

