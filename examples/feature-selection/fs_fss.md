## Feature Selection with Forward Sequential Selection (FSS)

Forward Sequential Selection starts from an empty set and iteratively adds the feature that most improves a chosen criterion (e.g., validation accuracy), stopping by a rule or when performance no longer improves.

This example uses FSS (forward sequential selection) to build a subset of features by adding, at each step, the feature that most improves the evaluation criterion.

Prerequisites
- R packages: daltoolbox, daltoolboxdp


``` r
# Feature Selection

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
# FSS - step by step

# 1) Fit the selector with target "Species"
myfeature <- fit(fs_fss("Species"), iris)

# 2) View selected features
print(myfeature$features)
```

```
## [1] "Sepal.Length" "Petal.Length" "Petal.Width"
```

``` r
# 3) Transform data to keep selected features + target
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Sepal.Length Petal.Length Petal.Width Species
## 1          5.1          1.4         0.2  setosa
## 2          4.9          1.4         0.2  setosa
## 3          4.7          1.3         0.2  setosa
## 4          4.6          1.5         0.2  setosa
## 5          5.0          1.4         0.2  setosa
## 6          5.4          1.7         0.4  setosa
```

References
- Whitney, A. W. (1971). A direct method of nonparametric measurement selection. IEEE Trans. Computers, 20(9), 1100–1103.

