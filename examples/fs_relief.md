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

### Relief


```r
myfeature <- fit(fs_relief("Species"), iris)
print(myfeature$features)
```

```
## [1] "Petal.Width"  "Petal.Length"
```

```r
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

