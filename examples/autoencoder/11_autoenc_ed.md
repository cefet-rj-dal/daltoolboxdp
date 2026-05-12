## Autoencoder (Encode-Decode) - Overview

The autoencoder jointly learns an encoder (p → k) and decoder (k → p) by minimizing reconstruction error. With sufficient capacity and regularization, the bottleneck enforces information compression so that reconstructions approximate inputs with low error.

This example shows an autoencoder that encodes and reconstructs the input. After training the reduction from p -> k dimensions, the model decodes back to p. The better the training, the closer the reconstruction is to the original (low reconstruction error).

Prerequisites
- Reticulate configured and Python with PyTorch installed
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2

Steps
1) Build time-series windows
2) Normalize data
3) Split into train and test
4) Train the AE (5 -> 3) and track losses
5) Reconstruct and compute metrics (R2, MAPE)


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
# Vanilla autoencoder transformation (encode-decode)

# Considering a dataset with $p$ numerical attributes.

# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensions the data is returned back to $p$ dimensions. The higher the autoencoder quality, the more similar is the output to the input.

# Installing packages
#install.packages("tspredit")
#install.packages("daltoolboxdp")
```


``` r
# Loading packages
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```


``` r
# Example dataset (series -> windows)
data(tsd)

sw_size <- 5
ts <- ts_data(tsd$y, sw_size)

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
set_example_seed()
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
# Train / test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```


``` r
# Training autoencoder (reduction 5 -> 3)
auto <- autoenc_ed(5, 3)
set_example_seed()
auto <- fit(auto, train)
```

Constructor configuration
- Fixed-epoch baseline: omit `epochs` to use the default value and keep `validation_strategy = "static"` with `stopping_rule = "none"`.
- Static early stopping: keep `validation_strategy = "static"` and choose `stopping_rule = "patience"`, `"sma"`, `"ema"`, or `"h"`.
- Dynamic early stopping: switch `validation_strategy = "dynamic"` and reuse the same stopping rules.
- The loss plot below always shows `train_loss`; it adds `val_loss` when validation is active.

Architecture variations
- Use `encoder_hidden_sizes` and `decoder_hidden_sizes` to control compression and reconstruction depth separately.
- When inputs are normalized to `[0, 1]`, `output_activation = "sigmoid"` can be useful; otherwise keep `"none"` to preserve the original linear decoder output.


``` r
# Loss curves
fit_loss <- data.frame(x = seq_along(auto$train_loss), train_loss = auto$train_loss)
if (!is.null(auto$val_loss) && length(auto$val_loss) > 0) {
  fit_loss$val_loss <- auto$val_loss
}

colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/11_autoenc_ed/unnamed-chunk-7-1.png)


``` r
# Testing: reconstruction of the test set
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
## [1,] 0.8338761 0.8604769 0.9019774 0.9323447 0.9258677
## [2,] 0.9181236 0.9136066 0.9742942 0.9924589 0.9724154
## [3,] 0.9684193 0.9366517 1.0060196 1.0205371 0.9859356
## [4,] 0.9826134 0.9356041 1.0024039 1.0116138 0.9681386
## [5,] 0.9674229 0.9112488 0.9653751 0.9744104 0.9217945
## [6,] 0.9212810 0.8662626 0.9016109 0.9079475 0.8484702
```


``` r
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
```

```
## [1] "t4 R2 test: 0.492766922565886 MAPE: 0.132009926318597"
## [1] "t3 R2 test: 0.907468867303464 MAPE: 0.0859221229428999"
## [1] "t2 R2 test: 0.999465116080585 MAPE: 0.0189727563552642"
## [1] "t1 R2 test: 0.966691645978666 MAPE: 0.108540735838356"
## [1] "t0 R2 test: 0.945838766531938 MAPE: 0.218136374491529"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.862446263692108 MAPE: 0.112716383189329"
```

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)

