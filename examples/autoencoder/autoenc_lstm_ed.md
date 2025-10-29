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
## [1,] 0.7791305
## [2,] 0.8382266
## [3,] 0.8796721
## [4,] 0.9046287
## [5,] 0.9129651
## [6,] 0.9040152
## 
## , , 2
## 
##           [,1]
## [1,] 0.8491536
## [2,] 0.8922189
## [3,] 0.9148911
## [4,] 0.9170530
## [5,] 0.8984872
## [6,] 0.8604644
## 
## , , 3
## 
##           [,1]
## [1,] 0.9154285
## [2,] 0.9580090
## [3,] 0.9768018
## [4,] 0.9712318
## [5,] 0.9409063
## [6,] 0.8876836
## 
## , , 4
## 
##           [,1]
## [1,] 0.9566564
## [2,] 0.9839603
## [3,] 0.9852595
## [4,] 0.9588034
## [5,] 0.9043209
## [6,] 0.8258265
## 
## , , 5
## 
##           [,1]
## [1,] 0.9551036
## [2,] 0.9659207
## [3,] 0.9531628
## [4,] 0.9144510
## [5,] 0.8491157
## [6,] 0.7611289
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
## [1] "t4 R2 test: 0.804374838819537 MAPE: 0.0737629086357512"
## [1] "t3 R2 test: 0.920300151497383 MAPE: 0.0821172508083947"
## [1] "t2 R2 test: 0.997721088169073 MAPE: 0.0162792967891201"
## [1] "t1 R2 test: 0.998924976088286 MAPE: 0.0299649571796719"
## [1] "t0 R2 test: 0.99265843231661 MAPE: 0.105510452853514"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.942795897378178 MAPE: 0.0615269732532903"
```

