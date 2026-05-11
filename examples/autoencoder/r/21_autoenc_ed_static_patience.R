# Installing packages
#install.packages("tspredit")
#install.packages("daltoolboxdp")

# Loading packages
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)

# Example dataset (series -> windows)
data(tsd)

sw_size <- 5
ts <- ts_data(tsd$y, sw_size)

preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)

# Static validation with patience-based early stopping
auto <- autoenc_ed(
  5, 3,
  epochs = 300L,
  validation_strategy = "static",
  stopping_rule = "patience",
  patience = 20L,
  val_ratio = 0.2
)
auto <- fit(auto, train)

# Effective training duration
print(length(auto$train_loss))

# Training and validation curves
fit_loss <- data.frame(x = seq_along(auto$train_loss), train_loss = auto$train_loss)
if (!is.null(auto$val_loss) && length(auto$val_loss) > 0) {
  fit_loss$val_loss <- auto$val_loss
}

colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)

# Testing: reconstruction of the test set
result <- transform(auto, test)
result <- as.data.frame(result)
names(result) <- names(test)
print(head(result))
