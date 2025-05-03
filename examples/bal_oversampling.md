## Feature Selection


``` r
# DALToolbox Data Preprocessing
# version 1.0.777

#loading DAL
library(daltoolbox) 
```

```
## 
## Attaching package: 'daltoolbox'
```

```
## The following objects are masked from 'package:daltoolboxdp':
## 
##     cla_knn, cla_mlp, cla_nb, cla_rf
```

```
## The following object is masked from 'package:base':
## 
##     transform
```

``` r
library(daltoolboxdp)
```

### Example
General function for exploring feature selection methods


``` r
iris <- datasets::iris
data(iris)
mod_iris <- iris[c(1:50,51:71,101:111),]
table(mod_iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```

### oversampling


``` r
bal <- bal_oversampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         50         42         44
```

