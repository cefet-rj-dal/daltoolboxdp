## Gradient Boosting Classifier - Overview

Gradient Boosting builds an additive ensemble of shallow trees by sequentially fitting each tree to the negative gradient (residuals) of a differentiable loss. Learning rate and tree depth control model complexity and generalization.

This example uses Gradient Boosting (scikit-learn via reticulate) to classify the Iris dataset.
Workflow: split train/test, train, predict class scores, and evaluate (classification metrics).

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python accessible via reticulate (scikit-learn installed)


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Gradient Boosting Classifier

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
# Training and evaluation with Gradient Boosting

slevels <- levels(iris$Species)                 # target variable levels

set.seed(1)
sr <- sample_random()                           # stratified random sampling
sr <- train_test(sr, iris)                      # split data
iris_train <- sr$train
iris_test <- sr$test

# Numeric encoding of the target for scikit-learn (keeping Species as original target)
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

# 1) Train
model <- skcla_gb("species_encoded", slevels)
set_example_seed()
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)
head(train_prediction)
```

```
##         setosa   versicolor    virginica
## 1 1.194090e-05 9.999487e-01 3.937048e-05
## 2 1.145890e-06 4.092976e-06 9.999948e-01
## 3 9.999704e-01 2.750743e-05 2.079094e-06
## 4 9.999704e-01 2.750743e-05 2.079094e-06
## 5 8.164130e-06 9.999706e-01 2.120526e-05
## 6 2.439474e-05 9.998952e-01 8.042047e-05
```

``` r
# 2) Evaluate on train
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 39 81  0  0         1      1           1           1  1
```

``` r
# 3) Evaluate on test
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)

test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```

References
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.

