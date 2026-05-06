## -----------------------------------------------------------------------------
# Installation (if needed)
#install.packages("daltoolboxdp")


## -----------------------------------------------------------------------------
library(daltoolbox)
library(daltoolboxdp)


## -----------------------------------------------------------------------------
# Loading Iris dataset
iris <- datasets::iris


## -----------------------------------------------------------------------------
# Training and evaluation with PyTorch MLP
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

model <- torch_cla_mlp(
  attribute = "Species",
  slevels = slevels,
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 100L
)

model <- fit(model, iris_train)
train_prediction <- predict(model, iris_train)

iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)


## -----------------------------------------------------------------------------
# Test prediction and evaluation
test_prediction <- predict(model, iris_test)

iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)


## -----------------------------------------------------------------------------
# Predicted probabilities
probabilities <- predict_proba.torch_cla_mlp(model, iris_test[, !names(iris_test) %in% "Species"])
head(probabilities)

