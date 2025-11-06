# Vanilla autoencoder transformation (encode-decode)

# Considering a dataset with $p$ numerical attributes.

# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensions the data is returned back to $p$ dimensions. The higher the autoencoder quality, the more similar is the output to the input.

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

ts_head(ts)

# Normalization (min-max by group)
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)

# Train / test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)

# Training autoencoder (reduction 5 -> 3)
auto <- autoenc_ed(5, 3)
auto <- fit(auto, train)

# Loss curves (train and validation)
fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss, val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)

# Testing: reconstruction of the test set
print(head(test))
result <- transform(auto, test)
print(head(result))

# Evaluating reconstruction quality: R2 and MAPE per attribute
result <- as.data.frame(result)
names(result) <- names(test)
r2 <- c()
mape <- c()
for (col in names(test)){
  r2_col <- cor(test[col], result[col])^2
  r2 <- append(r2, r2_col)
  mape_col <- mean((abs((result[col] - test[col]))/test[col])[[col]])
  mape <- append(mape, mape_col)
  print(paste(col, 'R2 test:', r2_col, 'MAPE:', mape_col))
}
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
