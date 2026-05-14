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
set_example_seed()
auto <- fit(auto, train)
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
## [1] 180
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

![plot of chunk unnamed-chunk-6](fig/22_autoenc_ed_dynamic_patience/unnamed-chunk-6-1.png)


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
## 1 0.7320997 0.8440774 0.9196408 0.9777486 0.9886138
## 2 0.8381691 0.9200779 0.9878805 1.0046357 0.9955263
## 3 0.9213054 0.9708138 1.0261103 0.9928624 0.9679638
## 4 0.9789316 0.9837417 1.0221833 0.9506516 0.9063077
## 5 1.0095520 0.9682120 0.9818302 0.8855848 0.8195416
## 6 1.0087428 0.9278623 0.9103274 0.7982647 0.7147818
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
## [1] "t4 R2 test: 0.991510430144083 MAPE: 0.0152782025498382"
## [1] "t3 R2 test: 0.981902922528159 MAPE: 0.0328720067486084"
## [1] "t2 R2 test: 0.999789694833044 MAPE: 0.0297439113710193"
## [1] "t1 R2 test: 0.997995318520696 MAPE: 0.0165330667484762"
## [1] "t0 R2 test: 0.99964734703793 MAPE: 0.00881027750877107"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.994169142612783 MAPE: 0.0206474929853426"
```

Notes
- Dynamic validation typically produces a noisier validation curve than the static case.
- To compare with a fixed validation split, see `21_autoenc_ed_static_patience`.

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
