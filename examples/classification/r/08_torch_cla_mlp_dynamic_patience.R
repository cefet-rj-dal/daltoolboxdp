source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Installation (if needed)
#install.packages("daltoolboxdp")

library(daltoolbox)
library(daltoolboxdp)

# Loading Iris dataset
iris <- datasets::iris

# Train/test split
slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Dynamic validation with patience-based early stopping
model <- torch_cla_mlp(
  attribute = "Species",
  slevels = slevels,
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 300L,
  validation_strategy = "dynamic",
  stopping_rule = "patience",
  patience = 20L,
  val_ratio = 0.2
)

set_example_seed()
model <- fit(model, iris_train)

# Training evaluation
train_prediction <- predict(model, iris_train)
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)

# Test evaluation
test_prediction <- predict(model, iris_test)
iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)

# Effective training duration
print(model$epochs_done)

# Training and validation curves
fit_loss <- data.frame(
  x = seq_along(model$train_loss_hist),
  train_loss = model$train_loss_hist
)

if (!is.null(model$val_loss_hist) && length(model$val_loss_hist) > 0) {
  fit_loss$val_loss <- model$val_loss_hist
}

colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)
