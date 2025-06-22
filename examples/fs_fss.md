
``` r
# Feature Selection

# installing packages

install.packages("daltoolboxdp")
```


``` r
# loading DAL
library(daltoolbox)
library(daltoolboxdp)
```



``` r
#General function for exploring feature selection methods

iris <- datasets::iris
```


``` r
# FSS

myfeature <- fit(fs_fss("Species"), iris)
print(myfeature$features)
```

```
## [1] "Sepal.Length" "Petal.Length" "Petal.Width"
```

``` r
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

