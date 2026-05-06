# Time Series Regression - LSTM

# Installing packages (if needed)

#install.packages("daltoolboxdp")

# Loading the packages
library(daltoolbox)
library(daltoolboxdp)
library(tspredit)
library(ggplot2)

# Series for study and sliding windows

data(tsd)
ts <- ts_data(tsd$y, 10)
ts_head(ts, 3)

# Series visualization
plot_ts(x = tsd$x, y = tsd$y) + theme(text = element_text(size = 16))

# Train-test split and projection (X, y)

samp <- ts_sample(ts, test_size = 5)
io_train <- ts_projection(samp$train)
io_test <- ts_projection(samp$test)

# Training the LSTM model

model <- ts_lstm(ts_norm_gminmax(), input_size = 4)
model <- fit(model, x = io_train$input, y = io_train$output)

# Fit evaluation (train)

adjust <- predict(model, io_train$input)
adjust <- as.vector(adjust)
output <- as.vector(io_train$output)
ev_adjust <- evaluate(model, output, adjust)
ev_adjust$mse

# Forecast on test set

steps_ahead <- 1
prediction <- predict(model, x = io_test$input, steps_ahead = steps_ahead)
prediction <- as.vector(prediction)

output <- as.vector(io_test$output)
if (steps_ahead > 1)
  output <- output[1:steps_ahead]

print(sprintf("%.2f, %.2f", output, prediction))

# Test evaluation

ev_test <- evaluate(model, output, prediction)
print(head(ev_test$metrics))
print(sprintf("smape: %.2f", 100 * ev_test$metrics$smape))

# Plot results

yvalues <- c(io_train$output, io_test$output)
plot_ts_pred(y = yvalues, yadj = adjust, ypre = prediction) + theme(text = element_text(size = 16))

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
