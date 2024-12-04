## Feature Selection


```r
# DALToolbox Data Preprocessing
# version 1.0.777

#loading DAL
library(daltoolbox) 
library(daltoolboxdp)
```

### Example
General function for exploring feature selection methods


```r
iris <- datasets::iris
```

### Lasso


```r
myfeature <- fit(fs_lasso("Species"), iris)
print(myfeature$features)
```

```
## [1] "Sepal.Width"  "Petal.Length" "Petal.Width"
```

```r
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

