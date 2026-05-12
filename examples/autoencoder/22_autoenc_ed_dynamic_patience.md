## 22. Autoencoder (Encode-Decode) with Dynamic Validation and Patience

This example keeps the same simple dense autoencoder architecture, but redraws the validation split at each epoch. It is useful when a single fixed holdout seems too brittle for the available data.

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
# Dynamic validation with patience-based early stopping
auto <- autoenc_ed(
  5, 3,
  epochs = 300L,
  validation_strategy = "dynamic",
  stopping_rule = "patience",
  patience = 20L,
  val_ratio = 0.2
)
```

```
## Warning: restarting interrupted promise evaluation
```

```
## Warning: internal error 1 in R_decompress1 with libdeflate
```

```
## Error:
## ! lazy-load database 'C:/R/R-4.5.0/library/daltoolboxdp/R/daltoolboxdp.rdb' is corrupt
```

``` r
set_example_seed()
auto <- fit(auto, train)
```

```
## Error:
## ! object 'auto' not found
```

Training configuration
- `validation_strategy = "dynamic"` redraws the validation split at each epoch.
- `stopping_rule = "patience"` stops training when recent validation values stop improving enough.
- `epochs = 300L` is only a ceiling; the realized duration depends on early stopping.


``` r
# Effective training duration
print(length(auto$train_loss))
```

```
## Error:
## ! object 'auto' not found
```


``` r
# Training and validation curves
fit_loss <- data.frame(x = seq_along(auto$train_loss), train_loss = auto$train_loss)
```

```
## Error:
## ! object 'auto' not found
```

``` r
if (!is.null(auto$val_loss) && length(auto$val_loss) > 0) {
  fit_loss$val_loss <- auto$val_loss
}
```

```
## Error:
## ! object 'auto' not found
```

``` r
colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
```

```
## Error:
## ! object 'fit_loss' not found
```

``` r
grf <- plot_series(fit_loss, colors = colors)
```

```
## Error:
## ! object 'fit_loss' not found
```

``` r
plot(grf)
```

```
## Error:
## ! object 'grf' not found
```


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
```

```
## Error:
## ! object 'auto' not found
```

``` r
result <- as.data.frame(result)
```

```
## Error:
## ! object 'result' not found
```

``` r
names(result) <- names(test)
```

```
## Error:
## ! object 'result' not found
```

``` r
print(head(result))
```

```
## Error:
## ! object 'result' not found
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
## Error:
## ! object 'result' not found
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## Warning in mean.default(r2): argument is not numeric or logical: returning NA
```

```
## Warning in mean.default(mape): argument is not numeric or logical: returning NA
```

```
## [1] "Means R2 test: NA MAPE: NA"
```

Notes
- Dynamic validation typically produces a noisier validation curve than the static case.
- To compare with a fixed validation split, see `21_autoenc_ed_static_patience`.

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
