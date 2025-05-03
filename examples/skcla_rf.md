## Random Forest Classifier


``` r
# DALToolbox Data Preprocessing
# version 1.0.777

#loading DAL
library(daltoolbox) 
library(daltoolboxdp)
```

### Example
#General function for exploring Random Forest classifier


``` r
iris <- datasets::iris
```

### Random Forest


``` r
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_rf("species_encoded", slevels)
model <- fit(model, iris_train_label)
```

```
## Column types: Sepal.Length       float64
## Sepal.Width        float64
## Petal.Length       float64
## Petal.Width        float64
## species_encoded      int32
## dtype: object
## Shape of data: (120, 5)
```

``` r
train_prediction <- predict(model, iris_train_label)
```

```
##      Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
## 0             5.8          2.7           4.1          1.0
## 1             6.4          2.8           5.6          2.1
## 2             4.4          3.2           1.3          0.2
## 3             4.3          3.0           1.1          0.1
## 4             7.0          3.2           4.7          1.4
## ..            ...          ...           ...          ...
## 115           5.0          3.4           1.6          0.4
## 116           6.0          2.2           4.0          1.0
## 117           5.6          2.8           4.9          2.0
## 118           6.4          3.2           5.3          2.3
## 119           6.7          3.1           4.4          1.4
## 
## [120 rows x 4 columns]
```

``` r
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 39 81  0  0         1      1           1           1  1
```

``` r
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)
```

```
##     Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
## 0            4.6          3.1           1.5          0.2
## 1            5.0          3.6           1.4          0.2
## 2            5.0          3.4           1.5          0.2
## 3            4.4          2.9           1.4          0.2
## 4            5.4          3.7           1.5          0.2
## 5            5.7          4.4           1.5          0.4
## 6            4.7          3.2           1.6          0.2
## 7            5.0          3.2           1.2          0.2
## 8            4.8          3.0           1.4          0.3
## 9            5.1          3.8           1.6          0.2
## 10           5.3          3.7           1.5          0.2
## 11           6.4          3.2           4.5          1.5
## 12           5.5          2.3           4.0          1.3
## 13           6.3          3.3           4.7          1.6
## 14           5.9          3.0           4.2          1.5
## 15           5.6          3.0           4.5          1.5
## 16           6.2          2.2           4.5          1.5
## 17           6.1          2.8           4.0          1.3
## 18           5.7          2.6           3.5          1.0
## 19           5.5          2.5           4.0          1.3
## 20           5.7          3.0           4.2          1.2
## 21           5.7          2.9           4.2          1.3
## 22           5.1          2.5           3.0          1.1
## 23           7.1          3.0           5.9          2.1
## 24           4.9          2.5           4.5          1.7
## 25           6.7          2.5           5.8          1.8
## 26           6.5          3.0           5.5          1.8
## 27           6.1          3.0           4.9          1.8
## 28           6.3          3.4           5.6          2.4
## 29           6.0          3.0           4.8          1.8
```

``` r
iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```
