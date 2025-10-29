## Class Balance: Oversampling

Random oversampling increases the representation of the minority class by replicating existing minority samples until a desired class balance is reached. It is simple and effective but may increase overfitting by duplicating examples.

This example shows how to handle class imbalance by applying oversampling (increase the minority class) on an imbalanced subset of the Iris dataset.

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
mod_iris <- iris[c(1:50,51:71,101:111),]   # subset with imbalanced classes
table(mod_iris$Species)                     # original distribution
```

```
## 
##     setosa versicolor  virginica 
##         50         21         11
```


``` r
# Oversampling - increase the minority class to balance
bal <- bal_oversampling('Species')
bal <- daltoolbox::fit(bal, mod_iris)
adjust_iris <- daltoolbox::transform(bal, mod_iris)
table(adjust_iris$Species)                   # distribution after oversampling
```

```
## 
##     setosa versicolor  virginica 
##         50         42         44
```

References
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.

