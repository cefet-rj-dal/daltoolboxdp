# Installation (if needed)
#install.packages("daltoolboxdp")

library(daltoolbox)
library(daltoolboxdp)

data <- mtcars

set.seed(1)
idx <- sample(seq_len(nrow(data)), size = floor(0.8 * nrow(data)))
train <- data[idx, ]
test <- data[-idx, ]

# Fit the regressor
model <- torch_reg_mlp(
  attribute = "mpg",
  input_size = ncol(data) - 1L,
  hidden_sizes = c(16L, 8L),
  epochs = 100L
)

model <- fit(model, train)

# Test prediction
prediction <- predict(model, test)
head(prediction)

# Regression evaluation
test_eval <- evaluate(model, test$mpg, prediction)
print(test_eval$metrics)
