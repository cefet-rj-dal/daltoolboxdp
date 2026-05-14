## SVM (Support Vector Machine) Classifier

Support Vector Machines find a maximum-margin hyperplane separating classes in a (possibly) high-dimensional feature space. Using kernels, SVMs implicitly map inputs to a feature space where linear separation is easier. The margin is controlled by a regularization parameter that trades off margin width and classification errors.

This example uses SVM (scikit-learn via reticulate) to classify the Iris dataset. Workflow: split train/test, train, predict class scores, and evaluate.

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
# Training and evaluation with SVM

slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Numeric encoding of the target for scikit-learn
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_svc("species_encoded", slevels, probability = TRUE)
set_example_seed()
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)
head(train_prediction)
```

```
##        setosa  versicolor   virginica
## 1 0.014935939 0.981461148 0.003602913
## 2 0.007765275 0.005611677 0.986623049
## 3 0.972596071 0.017142937 0.010260992
## 4 0.975868550 0.014102181 0.010029269
## 5 0.016318477 0.924225228 0.059456295
## 6 0.010142411 0.937662715 0.052194874
```

``` r
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1    0.975 39 81  0  0         1      1           1           1  1
```

``` r
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
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20, 273–297.

