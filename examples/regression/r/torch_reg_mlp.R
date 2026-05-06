## -----------------------------------------------------------------------------
# Regression MLP with PyTorch

# installation
#install.packages("daltoolboxdp")

# loading DAL
library(daltoolbox)
library(daltoolboxdp)


## -----------------------------------------------------------------------------
# Dataset for regression analysis

library(MASS)
data(Boston)
print(t(sapply(Boston, class)))
head(Boston)


## -----------------------------------------------------------------------------
# for performance, you can convert to matrix
Boston <- as.matrix(Boston)


## -----------------------------------------------------------------------------
# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, Boston)
boston_train <- sr$train
boston_test <- sr$test


## -----------------------------------------------------------------------------
# Training

model <- torch_reg_mlp(
  attribute = "medv",
  input_size = ncol(Boston) - 1L,
  hidden_sizes = c(16L, 8L),
  epochs = 100L
)
model <- fit(model, boston_train)


## -----------------------------------------------------------------------------
# Model adjustment

train_prediction <- predict(model, boston_train)
boston_train_predictand <- boston_train[, "medv"]
train_eval <- evaluate(model, boston_train_predictand, train_prediction)
print(train_eval$metrics)


## -----------------------------------------------------------------------------
# Test

test_prediction <- predict(model, boston_test)
boston_test_predictand <- boston_test[, "medv"]
test_eval <- evaluate(model, boston_test_predictand, test_prediction)
print(test_eval$metrics)

