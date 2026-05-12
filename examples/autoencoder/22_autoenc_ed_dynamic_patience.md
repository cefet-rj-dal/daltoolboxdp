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
## [1] 186
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
## 1 0.7264100 0.8452213 0.9181046 0.9842790 0.9971019
## 2 0.8324886 0.9270588 0.9860609 1.0081099 1.0018766
## 3 0.9164250 0.9853894 1.0223109 0.9972882 0.9758188
## 4 0.9738216 1.0060178 1.0206773 0.9509513 0.9091663
## 5 1.0052170 0.9940958 0.9798200 0.8834693 0.8166361
## 6 1.0054078 0.9555614 0.9074836 0.7935561 0.7073630
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
## [1] "t4 R2 test: 0.993585006408508 MAPE: 0.0113162341669564"
## [1] "t3 R2 test: 0.993824613808065 MAPE: 0.0170454524092029"
## [1] "t2 R2 test: 0.999828254587985 MAPE: 0.0292738471761549"
## [1] "t1 R2 test: 0.996855306669369 MAPE: 0.0170665366110415"
## [1] "t0 R2 test: 0.99979714758425 MAPE: 0.0156558491937935"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.996778065811635 MAPE: 0.0180715839114298"
```

Notes
- Dynamic validation typically produces a noisier validation curve than the static case.
- To compare with a fixed validation split, see `21_autoenc_ed_static_patience`.

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
