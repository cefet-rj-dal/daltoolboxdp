## LSTM Autoencoder (encode-decode)

This example demonstrates the use of an LSTM-based Autoencoder to encode windows of a time series (p -> k) and reconstruct them (k -> p). This allows evaluation of reconstruction quality.

Prerequisites
- Python with PyTorch accessible via reticulate
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2


``` r
# Installing example dependencies (if needed)
#install.packages("tspredit")
#install.packages("daltoolboxdp")
```


``` r
# Loading required packages
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```


``` r
# Example dataset (series -> windows)
data(tsd)

sw_size <- 5                      # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)     # convert series into windows with p columns

ts_head(ts)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
## [4,] 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859
## [5,] 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974
## [6,] 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
```


``` r
# Normalization (min-max by group)
preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

ts_head(ts)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.5004502 0.6243512 0.7405486 0.8418178 0.9218625
## [2,] 0.6243512 0.7405486 0.8418178 0.9218625 0.9757058
## [3,] 0.7405486 0.8418178 0.9218625 0.9757058 1.0000000
## [4,] 0.8418178 0.9218625 0.9757058 1.0000000 0.9932346
## [5,] 0.9218625 0.9757058 1.0000000 0.9932346 0.9558303
## [6,] 0.9757058 1.0000000 0.9932346 0.9558303 0.8901126
```


``` r
# Train/test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```


``` r
# Creating the LSTM autoencoder (encode-decode): 5 -> 3 -> 5 dimensions
auto <- autoenc_lstm_ed(5, 3, num_epochs = 1500)

# Training the model
auto <- fit(auto, train)
```


``` r
# Learning curves (train and validation loss per epoch)
fit_loss <- data.frame(
  x = 1:length(auto$train_loss),
  train_loss = auto$train_loss,
  val_loss = auto$val_loss
)
grf <- plot_series(fit_loss, colors = c('Blue', 'Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_lstm_ed/unnamed-chunk-7-1.png)


``` r
# Testing the autoencoder (reconstruction)
# Show samples from the test set and the reconstruction (p columns)
print(head(test))
```

```
##          t4        t3        t2        t1        t0
## 1 0.7258342 0.8294719 0.9126527 0.9702046 0.9985496
## 2 0.8294719 0.9126527 0.9702046 0.9985496 0.9959251
## 3 0.9126527 0.9702046 0.9985496 0.9959251 0.9624944
## 4 0.9702046 0.9985496 0.9959251 0.9624944 0.9003360
## 5 0.9985496 0.9959251 0.9624944 0.9003360 0.8133146
## 6 0.9959251 0.9624944 0.9003360 0.8133146 0.7068409
```

``` r
result <- transform(auto, test)
print(head(result))
```

```
## , , 1
## 
##           [,1]
## [1,] 0.7175261
## [2,] 0.8285640
## [3,] 0.9068478
## [4,] 0.9522040
## [5,] 0.9727197
## [6,] 0.9751367
## 
## , , 2
## 
##           [,1]
## [1,] 0.8633706
## [2,] 0.9339829
## [3,] 0.9733570
## [4,] 0.9804008
## [5,] 0.9565164
## [6,] 0.8995060
## 
## , , 3
## 
##           [,1]
## [1,] 0.9054876
## [2,] 0.9605258
## [3,] 0.9900403
## [4,] 0.9925365
## [5,] 0.9671333
## [6,] 0.9096718
## 
## , , 4
## 
##           [,1]
## [1,] 0.9326348
## [2,] 0.9643930
## [3,] 0.9750820
## [4,] 0.9629625
## [5,] 0.9244353
## [6,] 0.8522531
## 
## , , 5
## 
##           [,1]
## [1,] 0.9470259
## [2,] 0.9518526
## [3,] 0.9412059
## [4,] 0.9130806
## [5,] 0.8608191
## [6,] 0.7737978
```


``` r
# Reconstruction metrics per column: R2 and MAPE
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
```

```
## [1] "t4 R2 test: 0.930416516523359 MAPE: 0.0275825605136812"
## [1] "t3 R2 test: 0.922320254714504 MAPE: 0.0778465665882138"
## [1] "t2 R2 test: 0.998699682024783 MAPE: 0.0104890034070287"
## [1] "t1 R2 test: 0.989488654175949 MAPE: 0.0345787485896423"
## [1] "t0 R2 test: 0.982493789443021 MAPE: 0.0612805721626366"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.964683779376323 MAPE: 0.0423554902522405"
```

### Method

An LSTM encoder summarizes each window into a compact state that a decoder uses to reconstruct the original sequence. Training minimizes reconstruction loss, encouraging the latent state to retain information about temporal dynamics in the window.

### References
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
