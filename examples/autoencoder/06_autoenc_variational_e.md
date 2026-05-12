## Variational Autoencoder (encode)

Variational Autoencoders learn a probabilistic encoder that outputs parameters of a latent distribution (e.g., mean and variance) and a decoder that reconstructs from latent samples. The loss combines reconstruction error and a KL divergence that regularizes the latent distribution toward a prior.

This example uses a Variational Autoencoder (VAE) to learn latent representations of time-series windows. The VAE reduces from p to k dimensions and regularizes the latent space to approximate a target distribution (e.g., standard normal) via a KL term.

Prerequisites
- Python with PyTorch accessible via reticulate
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2

Quick notes
- Loss: reconstruction + KL divergence between the latent distribution and the prior.
- Useful for generating continuous, well-behaved representations in latent space.


``` r
source(url("https://raw.githubusercontent.com/cefet-rj-dal/daltoolboxdp/main/examples/seed.R"))
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
# Train/test split
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```


``` r
# Creating the VAE: reduce from 5 -> 3 dimensions (p -> k)
# - the default number of epochs is used
auto <- autoenc_variational_e(5, 3)

# Training the model
set_example_seed()
auto <- fit(auto, train)
```

Constructor configuration
- Fixed-epoch baseline: omit `epochs` to use the default value and keep `validation_strategy = "static"` with `stopping_rule = "none"`.
- Static early stopping: keep `validation_strategy = "static"` and choose `stopping_rule = "patience"`, `"sma"`, `"ema"`, or `"h"`.
- Dynamic early stopping: switch `validation_strategy = "dynamic"` and reuse the same stopping rules.
- The loss plot below always shows `train_loss`; it adds `val_loss` when validation is active.

Architecture variations
- The original VAE is recovered with `encoder_hidden_sizes = c(64L, 32L)`.
- Deeper VAEs can be built with larger stacks such as `c(128L, 64L, 32L)`.
- `reconstruction_loss = "bce"` works well with normalized inputs and `output_activation = "sigmoid"`.
- `reconstruction_loss = "mse"` is often preferable for unconstrained real-valued windows.


``` r
# Learning curves
fit_loss <- data.frame(
  x = seq_along(auto$train_loss),
  train_loss = auto$train_loss
)
if (!is.null(auto$val_loss) && length(auto$val_loss) > 0) {
  fit_loss$val_loss <- auto$val_loss
}
colors <- if ("val_loss" %in% names(fit_loss)) c("Blue", "Orange") else c("Blue")
grf <- plot_series(fit_loss, colors = colors)
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/06_autoenc_variational_e/unnamed-chunk-7-1.png)


``` r
# Testing the VAE (encoding)
# Show samples from the test set and the encoding (k columns)
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
##            [,1]       [,2]       [,3]        [,4]          [,5]          [,6]
## [1,] -0.1587287 -0.2147984 -0.2149044 -0.02928344 -1.133420e-03  0.0001546256
## [2,] -0.1840510 -0.2540534 -0.2568787 -0.03164933  3.829911e-03 -0.0018734634
## [3,] -0.1951965 -0.2730957 -0.2802375 -0.03364404  6.409667e-03 -0.0035988595
## [4,] -0.1882526 -0.2688529 -0.2819935 -0.03513510  5.773678e-03 -0.0045520756
## [5,] -0.1660521 -0.2429923 -0.2612343 -0.03403335  3.712982e-03 -0.0059001483
## [6,] -0.1323544 -0.1988099 -0.2207219 -0.03155528  4.784018e-05 -0.0069676861
```

References
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.

