# Installing example dependencies (if needed)
#install.packages("tspredit")
#install.packages("daltoolboxdp")

# Loading required packages
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)

# Example dataset (series -> windows)
data(tsd)

sw_size <- 5                      # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)     # convert series into windows with p columns

ts_head(ts)                       # preview first rows

# Normalization (min-max by group)
# Keeps each column (window step) on the same [0,1] scale
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)

# Train/test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)

# Creating the adversarial autoencoder: reduce from 5 -> 3 dimensions (p -> k)
# - batch_size: training batch size per step
# - num_epochs: number of training epochs
auto <- autoenc_adv_e(5, 3, batch_size = 3, num_epochs = 1500)

# Training the model on the train set
auto <- fit(auto, train)

# Learning curves (train and validation loss per epoch)
fit_loss <- data.frame(
  x = 1:length(auto$train_loss),
  train_loss = auto$train_loss,
  val_loss = auto$val_loss
)
grf <- plot_series(fit_loss, colors = c('Blue', 'Orange'))
plot(grf)

# Testing the autoencoder (encoding only)
# Show samples from the test set and the resulting encoding (k columns)
print(head(test))
result <- transform(auto, test)
print(head(result))
