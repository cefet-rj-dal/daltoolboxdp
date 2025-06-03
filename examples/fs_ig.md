
``` r
# Feature Selection

# installing packages

install.packages("daltoolboxdp")
```

```

```


``` r
# loading DAL
library(daltoolbox)
library(daltoolboxdp)
```



``` r
# General function for exploring feature selection methods
 
iris <- datasets::iris
```


``` r
# IG

myfeature <- fit(fs_ig("Species"), iris)
print(myfeature$features)
```

```
## [1] "Petal.Width"  "Petal.Length"
```

``` r
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

