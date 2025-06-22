
``` r
# Feature Selection

# installing packages

#install.packages("daltoolboxdp")
```


``` r
# loading DAL
library(daltoolbox)
library(daltoolboxdp)
```


``` r
# General function for exploring feature selection methods

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


``` r
# subsampling

bal <- bal_subsampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)
```

```
## 
##     setosa versicolor  virginica 
##         11         11         11
```

