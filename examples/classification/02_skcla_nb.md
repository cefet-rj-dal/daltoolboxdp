## Naive Bayes Classifier

Naive Bayes applies Bayes’ theorem under a conditional independence assumption: features are assumed independent given the class. Class-conditional likelihoods (e.g., Gaussian) are estimated per class, then combined with class priors to compute posterior probabilities used for classification.

This example uses Naive Bayes (scikit-learn via reticulate) to classify the Iris dataset.
Workflow: split train/test, train, predict, and evaluate.

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
# Training and evaluation with Naive Bayes

slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Numeric encoding of the target for scikit-learn
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_nb("species_encoded", slevels)
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
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.

