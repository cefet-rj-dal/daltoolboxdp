## Variational Autoencoder (encode-decode)

The VAE uses an encoder to parameterize a latent distribution and a decoder to reconstruct inputs from latent samples. Training optimizes an evidence lower bound with reconstruction and KL divergence terms, shaping a smooth, regularized latent space.

This example uses a Variational Autoencoder (VAE) to encode windows of a time series (p -> k) and reconstruct them (k -> p). We evaluate reconstruction quality at the end.

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
# Creating the VAE (encode-decode): 5 -> 3 -> 5 dimensions
auto <- autoenc_variational_ed(5, 3, num_epochs = 350)

# Training the model
auto <- fit(auto, train)
```


``` r
# Learning curves (total loss per epoch)
fit_loss <- data.frame(
  x = 1:length(auto$train_loss),
  train_loss = auto$train_loss,
  val_loss = auto$val_loss
)
grf <- plot_series(fit_loss, colors = c('Blue', 'Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_variational_ed/unnamed-chunk-7-1.png)


``` r
# Testing the VAE (reconstruction)
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
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.8554683 0.9080823 0.9313472 0.9249861 0.8901986
## [2,] 0.9117713 0.9510282 0.9660469 0.9607008 0.9338863
## [3,] 0.9359789 0.9671261 0.9780683 0.9735506 0.9515549
## [4,] 0.9355848 0.9665033 0.9773860 0.9726099 0.9501144
## [5,] 0.9116032 0.9491407 0.9634889 0.9570464 0.9286420
## [6,] 0.8448540 0.8931288 0.9144952 0.9050581 0.8667603
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
## [1] "t4 R2 test: 0.295189237328548 MAPE: 0.167941231099671"
## [1] "t3 R2 test: 0.88038909588862 MAPE: 0.0890507136936134"
## [1] "t2 R2 test: 0.992537237864864 MAPE: 0.0185665869262286"
## [1] "t1 R2 test: 0.958150989102786 MAPE: 0.133067342241115"
## [1] "t0 R2 test: 0.912343646777869 MAPE: 0.322132413671049"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.807722041392537 MAPE: 0.146151657526335"
```

References
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
