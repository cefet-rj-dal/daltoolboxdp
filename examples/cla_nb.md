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

model <- cla_nb("species_encoded", slevels)
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
## Error in nb_fit: The 'var_smoothing' parameter of GaussianNB must be a float in the range [0.0, inf). Got None instead.
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
## Error in nb_predict: This GaussianNB instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
```

```
## Warning in predict.cla_nb(model, iris_train_label): Prediction returned NULL or empty. Returning NA values.
```

``` r
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
```

``` r
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity  f1
## 1        0  0  0  0  0       NaN    NaN         NaN         NaN NaN
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
## Error in nb_predict: This GaussianNB instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
```

```
## Warning in predict.cla_nb(model, iris_test_label): Prediction returned NULL or empty. Returning NA values.
```

``` r
iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning -Inf
```

``` r
print(test_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity  f1
## 1        0  0  0  0  0       NaN    NaN         NaN         NaN NaN
```
