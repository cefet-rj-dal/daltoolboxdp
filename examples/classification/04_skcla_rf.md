## Random Forest Classifier - Overview

Random Forest is an ensemble of decision trees trained on bootstrap samples, where each split considers a random subset of features. This decorrelates trees and reduces variance. For classification, predictions are obtained by majority vote across trees.

This example uses Random Forest (scikit-learn via reticulate) to classify the Iris dataset.
Workflow: split train/test, train, predict class scores, and evaluate (classification metrics).

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python accessible via reticulate (scikit-learn installed)


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Install required packages (if not already installed)
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
# Training and evaluation with Random Forest

slevels <- levels(iris$Species)                 # target variable levels

set.seed(1)
sr <- sample_random()                           # stratified random sampling
sr <- train_test(sr, iris)                      # split data
iris_train <- sr$train
iris_test <- sr$test

# Create numeric label for scikit-learn (keeping "Species" as original target)
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

# 1) Train
model <- skcla_rf("species_encoded", slevels)
set_example_seed()
model <- fit(model, iris_train_label)

# 2) Evaluate on train
train_prediction <- predict(model, iris_train_label)
head(train_prediction)
```

```
##   setosa versicolor virginica
## 1   0.00       1.00      0.00
## 2   0.00       0.00      1.00
## 3   1.00       0.00      0.00
## 4   1.00       0.00      0.00
## 5   0.00       0.99      0.01
## 6   0.01       0.99      0.00
```

``` r
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
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

References
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

