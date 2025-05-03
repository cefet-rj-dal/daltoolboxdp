## Naive Bayes Classifier


``` r
# DALToolbox Data Preprocessing
# version 1.0.777

#loading DAL
library(daltoolbox) 
library(daltoolboxdp)
```

### Example
General function for exploring Naive Bayes classifier


``` r
iris <- datasets::iris
```

### Naive Bayes


``` r
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_nb("species_encoded", slevels)
model <- fit(model, iris_train_label)
```

```
## Fitting model with data dimensions: 120 x 5
```

```
## Target attribute: species_encoded
```

```
## X_train shape: (120, 4)
## y_train shape: (120,)
## X_train data type: float64
## y_train data type: int32
```

``` r
train_prediction <- predict(model, iris_train_label)
```

```
## Predicting with data dimensions: 120 x 4
```

```
## X_test shape: (120, 4)
## X_test data type: float64
```

``` r
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9583333 39 81  0  0         1      1           1           1  1
```

``` r
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)
```

```
## Predicting with data dimensions: 30 x 4
```

```
## X_test shape: (30, 4)
## X_test data type: float64
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
