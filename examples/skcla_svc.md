## Support Vector Machine Classifier


``` r
# DALToolbox Data Preprocessing
# version 1.0.777

#loading DAL
library(daltoolbox) 
library(daltoolboxdp)
```

### Example
#General function for exploring SVM classifier


``` r
iris <- datasets::iris
```

### SVM


``` r
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_svc("species_encoded", slevels)
model <- fit(model, iris_train_label)
```

```
## Error in skcla_svc_fit(obj$model, data, obj$attribute, obj$slevels): could not find function "skcla_svc_fit"
```

``` r
train_prediction <- predict(model, iris_train_label)
```

```
## Prediction input shape: (120, 4)
## Another error occurred: 'NoneType' object has no attribute 'predict'
```

``` r
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Warning in FUN(if (length(d.call) < 2L) newX[, 1] else array(newX[, 1L], : no non-missing arguments to max; returning -Inf
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
## Prediction input shape: (30, 4)
## Another error occurred: 'NoneType' object has no attribute 'predict'
```

``` r
iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

``` r
print(test_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity  f1
## 1        0  0  0  0  0       NaN    NaN         NaN         NaN NaN
```
