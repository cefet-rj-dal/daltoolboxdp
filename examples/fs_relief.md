## Feature Selection


``` r
# DALToolbox Data Preprocessing
# version 1.1.717

#loading DAL
library(daltoolbox) 
library(daltoolboxdp)
```

### Example
General function for exploring feature selection methods


``` r
iris <- datasets::iris
```

### Relief


``` r
myfeature <- fit(fs_relief("Species"), iris)
print(myfeature$features)
```

```
## [1] "Petal.Length" "Petal.Width"
```

``` r
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

