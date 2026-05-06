library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)

data(tsd)

sw_size <- 5
ts <- data.frame(unclass(ts_data(tsd$y, sw_size)))

split_at <- nrow(ts) - 5
train <- ts[1:split_at, ]
test <- ts[(split_at + 1):nrow(ts), ]

model <- ts_conv1d(
  preprocess = ts_norm_gminmax(),
  input_size = 4L,
  epochs = 100L
)

model <- fit(model, train[, 1:4], train[, 5])

# Training curves
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

# One-step-ahead prediction on the test windows
prediction <- predict(model, test[, 1:4])
print(prediction)
