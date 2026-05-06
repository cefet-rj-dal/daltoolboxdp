# Installation (if needed)
#install.packages("daltoolboxdp")

library(daltoolbox)
library(daltoolboxdp)

# Prepare Iris with a numeric target for the current wrapper
iris_torch <- data.frame(
  iris[, 1:4],
  species_encoded = as.integer(iris$Species)
)

set.seed(1)
idx <- sample(seq_len(nrow(iris_torch)), size = floor(0.8 * nrow(iris_torch)))
iris_train <- iris_torch[idx, ]
iris_test <- iris_torch[-idx, ]

# Fit the classifier
model <- torch_cla_mlp(
  attribute = "species_encoded",
  slevels = c(1L, 2L, 3L),
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 100L
)

model <- fit(model, iris_train)

# Predicted classes
prediction <- predict(model, iris_test)
head(prediction)

# Predicted probabilities
probabilities <- predict_proba.torch_cla_mlp(model, iris_test[, 1:4])
str(probabilities[1:3])
