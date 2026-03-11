## Multi-Layer Perceptron (MLP) Classifier

A Multi-Layer Perceptron is a feed-forward neural network with one or more hidden layers. Neurons apply an affine transformation followed by a nonlinearity. The network is trained to minimize a loss via backpropagation and gradient-based optimization, enabling nonlinear decision boundaries.

This example uses MLP (scikit-learn via reticulate) to classify the Iris dataset. Workflow: split train/test, train, predict, and evaluate.

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python accessible via reticulate (scikit-learn installed)


``` r
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
# Training and evaluation with MLP

slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Numeric encoding of the target for scikit-learn
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_mlp("species_encoded", slevels, max_iter = 1000)  # increase max_iter for convergence
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)

iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
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

iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 11 19  0  0         1      1           1           1  1
```

References
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536.

