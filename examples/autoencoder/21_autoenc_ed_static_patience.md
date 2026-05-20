## 21. Autoencoder (Encode-Decode) with Static Validation and Patience

This example keeps the simple dense autoencoder architecture, but changes the training regime to use a fixed validation split and patience-based early stopping. It is useful when reconstruction quality matters and you want a stable reference validation subset across epochs.

Prerequisites
- Reticulate configured and Python with PyTorch installed
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
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

preproc <- ts_norm_gminmax()
set_example_seed()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)

samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```


``` r
# Static validation with patience-based early stopping
auto <- autoenc_ed(
  5, 3,
  epochs = 300L,
  validation_strategy = "static",
  stopping_rule = "patience",
  patience = 20L,
  val_ratio = 0.2
)
set_example_seed()
auto <- fit(auto, train)
```

Training configuration
- `validation_strategy = "static"` keeps the same validation partition during the whole training run.
- `stopping_rule = "patience"` stops training when validation loss does not improve for a chosen number of epochs.
- `epochs = 300L` is only an upper bound; the effective number of epochs is determined by early stopping.


``` r
# Effective training duration
print(length(auto$train_loss))
```

```
## [1] 197
```


``` r
# Training and validation curves
fit_loss <- data.frame(x = seq_along(auto$train_loss), train_loss = auto$train_loss)
if (!is.null(auto$val_loss) && length(auto$val_loss) > 0) {
  fit_loss$val_loss <- auto$val_loss
}

colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)
```

![plot of chunk unnamed-chunk-6](fig/21_autoenc_ed_static_patience/unnamed-chunk-6-1.png)


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
result <- as.data.frame(result)
names(result) <- names(test)
print(head(result))
```

```
##          t4        t3        t2        t1        t0
## 1 0.7341869 0.8334709 0.9220324 0.9785485 0.9925448
## 2 0.8424433 0.9175313 0.9907760 1.0023085 0.9906574
## 3 0.9179645 0.9682583 1.0230904 0.9956362 0.9669256
## 4 0.9669870 0.9936911 1.0209659 0.9536166 0.9125043
## 5 0.9891561 0.9868173 0.9798599 0.8856937 0.8293657
## 6 0.9856901 0.9527866 0.9115646 0.7960237 0.7175472
```


``` r
# Evaluating reconstruction quality: R2 and MAPE per attribute
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
## [1] "t4 R2 test: 0.995771193005259 MAPE: 0.00850546220346787"
## [1] "t3 R2 test: 0.996981173679754 MAPE: 0.00833127050386143"
## [1] "t2 R2 test: 0.99985141304813 MAPE: 0.0275541071546"
## [1] "t1 R2 test: 0.998611201392783 MAPE: 0.0124415171574355"
## [1] "t0 R2 test: 0.998675939158132 MAPE: 0.0257244617264601"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.997978184056812 MAPE: 0.016511363749165"
```

Notes
- This setup is easier to interpret because all epochs are judged against the same validation subset.
- To compare with redrawn validation splits, see `22_autoenc_ed_dynamic_patience`.

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
