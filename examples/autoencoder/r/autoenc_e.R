# Vanilla autoencoder transformation (encode)

# Considering a dataset with $p$ numerical attributes. 

# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. 

# Installing packages
#install.packages("tspredit")
#install.packages("daltoolboxdp")

# Loading required packages
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)

# Example dataset (time series and windows)
data(tsd)

sw_size <- 5
ts <- ts_data(tsd$y, sw_size)

ts_head(ts)

# Normalization (min-max by group) to stabilize training
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)

# Train / test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)

# Creating and training the autoencoder (reduce from 5 to 3 dimensions)
auto <- autoenc_e(5, 3)
auto <- fit(auto, train)

# Visualizing loss curves (train/validation)
fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)

# Testing the autoencoder: encoding test examples
print(head(test))
result <- transform(auto, test)
print(head(result))
