source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Install required packages (if not already installed)
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(daltoolboxdp)

# Loading Iris dataset
iris <- datasets::iris

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
train_eval <- evaluate(model, iris_train[, "Species"], train_prediction)
print(train_eval$metrics)

# 3) Evaluate on test
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)

test_eval <- evaluate(model, iris_test[, "Species"], test_prediction)
print(test_eval$metrics)
