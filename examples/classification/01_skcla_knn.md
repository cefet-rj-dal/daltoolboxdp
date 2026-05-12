## K-Nearest Neighbors (KNN) Classifier

k-Nearest Neighbors is an instance-based classifier that assigns a class based on the majority label among the k closest training examples under a chosen distance metric. It is non-parametric and relies on local neighborhoods to infer class membership.

This example uses KNN (scikit-learn via reticulate) to classify the Iris dataset. Workflow: split train/test, train, predict, and evaluate.

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python accessible via reticulate (scikit-learn installed)


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Installation (if needed)
#install.packages("daltoolboxdp")
```


``` r
# Loading packages
library(daltoolbox)
library(daltoolboxdp)
```



``` r
# Loading Iris dataset
iris <- datasets::iris
```


``` r
# Training and evaluation with KNN

slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Numeric encoding of the target for scikit-learn
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_knn("species_encoded", slevels, n_neighbors = 1)
```

```
## Warning: internal error 1 in R_decompress1 with libdeflate
```

```
## Error:
## ! lazy-load database 'C:/R/R-4.5.0/library/daltoolboxdp/R/daltoolboxdp.rdb' is corrupt
```

``` r
set_example_seed()
model <- fit(model, iris_train_label)
```

```
## Error:
## ! object 'model' not found
```

``` r
train_prediction <- predict(model, iris_train_label)
```

```
## Error:
## ! object 'model' not found
```

``` r
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
```

```
## Error:
## ! object 'model' not found
```

``` r
print(train_eval$metrics)
```

```
## Error:
## ! object 'train_eval' not found
```

``` r
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)
```

```
## Error:
## ! object 'model' not found
```

``` r
iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
```

```
## Error:
## ! object 'model' not found
```

``` r
print(test_eval$metrics)
```

```
## Error:
## ! object 'test_eval' not found
```

References
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21–27.

