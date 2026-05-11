## 21. Autoencoder (Encode-Decode) with Static Validation and Patience

This example keeps the simple dense autoencoder architecture, but changes the training regime to use a fixed validation split and patience-based early stopping. It is useful when reconstruction quality matters and you want a stable reference validation subset across epochs.

Prerequisites
- Reticulate configured and Python with PyTorch installed
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2


``` r
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
## [1] 300
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
result <- transform(auto, test)
result <- as.data.frame(result)
names(result) <- names(test)
print(head(result))
```

```
##          t4        t3        t2        t1        t0
## 1 0.7262827 0.8230802 0.9121844 0.9636005 0.9935867
## 2 0.8251619 0.9052964 0.9696078 0.9902289 0.9902962
## 3 0.9040825 0.9652139 0.9996211 0.9901654 0.9586834
## 4 0.9657982 0.9970065 0.9988040 0.9571426 0.8954175
## 5 0.9962200 0.9944488 0.9606155 0.9002054 0.8145471
## 6 0.9940477 0.9632283 0.8936421 0.8173189 0.7103570
```

Notes
- This setup is easier to interpret because all epochs are judged against the same validation subset.
- To compare with redrawn validation splits, see `22_autoenc_ed_dynamic_patience`.

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
