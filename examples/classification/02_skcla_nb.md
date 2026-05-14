## Naive Bayes Classifier

Naive Bayes applies Bayes’ theorem under a conditional independence assumption: features are assumed independent given the class. Class-conditional likelihoods (e.g., Gaussian) are estimated per class, then combined with class priors to compute posterior probabilities used for classification.

This example uses Naive Bayes (scikit-learn via reticulate) to classify the Iris dataset.
Workflow: split train/test, train, predict class scores, and evaluate.

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
set_example_seed()
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)
head(train_prediction)
```

```
##          setosa   versicolor    virginica
## 1  1.839907e-54 9.999902e-01 9.845615e-06
## 2 3.388899e-169 1.736455e-05 9.999826e-01
## 3  1.000000e+00 6.966668e-19 2.133782e-27
## 4  1.000000e+00 6.808172e-20 1.595997e-28
## 5  5.230229e-94 8.598383e-01 1.401617e-01
## 6  2.602904e-84 9.951269e-01 4.873098e-03
```

``` r
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
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

test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9666667 11 19  0  0         1      1           1           1  1
```

References
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.

