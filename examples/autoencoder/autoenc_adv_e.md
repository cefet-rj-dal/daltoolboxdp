## Adversarial Autoencoder (encode)

Adversarial Autoencoders augment the autoencoder objective with an adversarial discriminator in latent space. The encoder is trained to both reconstruct inputs (via the decoder) and produce latent codes whose distribution matches a desired prior, improving latent structure.

This example shows how to train and use an Adversarial Autoencoder (AAE) to encode windows of a time series, reducing from p to k dimensions while imposing a desired distribution in the latent space via an adversarial discriminator.

Prerequisites
- Python with PyTorch accessible via reticulate
- R packages: daltoolbox, tspredit, daltoolboxdp, ggplot2

Quick notes
- Architecture: encoder + decoder, with a discriminator in the latent space for adversarial regularization.
- Goal: learn compact representations (k) that preserve information from the original windows (p).
- Important hyperparameters: `epochs`, `batch_size`, learning rate defined internally.


``` r
# Installing example dependencies (if needed)
#install.packages("tspredit")
#install.packages("daltoolboxdp")
```


``` r
# Loading required packages
library(daltoolbox)
```

```
## Warning: package 'daltoolbox' was built under R version 4.5.3
```

```
## 
## Attaching package: 'daltoolbox'
```

```
## The following object is masked from 'package:base':
## 
##     transform
```

``` r
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```

```
## Warning: package 'ggplot2' was built under R version 4.5.3
```


``` r
# Example dataset (series -> windows)
data(tsd)

sw_size <- 5                      # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)     # convert series into windows with p columns

ts_head(ts)                       # preview first rows
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
# Keeps each column (window step) on the same [0,1] scale
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
# Creating the adversarial autoencoder: reduce from 5 -> 3 dimensions (p -> k)
# - batch_size: training batch size per step
# - the default number of epochs is used
auto <- autoenc_adv_e(5, 3, batch_size = 3)

# Training the model on the train set
auto <- fit(auto, train)
```

Constructor configuration
- Fixed-epoch baseline: omit `epochs` to use the default value and keep `validation_strategy = "static"` with `stopping_rule = "none"`.
- Static early stopping: keep `validation_strategy = "static"` and choose `stopping_rule = "patience"`, `"sma"`, `"ema"`, or `"h"`.
- Dynamic early stopping: switch `validation_strategy = "dynamic"` and reuse the same stopping rules.
- The loss plot below always shows `train_loss`; it adds `val_loss` when validation is active.

Architecture variations
- `encoder_hidden_sizes`, `decoder_hidden_sizes`, and `discriminator_hidden_sizes` let you decouple the three adversarial sub-networks.
- `latent_prior_scale` controls how spread out the target latent prior is.
- `lr_encoder`, `lr_decoder`, `lr_generator`, and `lr_discriminator` can override the base `learning_rate` when the adversarial game becomes unstable.


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

![plot of chunk unnamed-chunk-7](fig/autoenc_adv_e/unnamed-chunk-7-1.png)


``` r
# Testing the autoencoder (encoding only)
# Show samples from the test set and the resulting encoding (k columns)
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
##          [,1]     [,2]     [,3]
## [1,] 2.949961 7.815766 3.035430
## [2,] 3.085739 8.613660 3.037390
## [3,] 3.162011 9.008178 3.056101
## [4,] 3.174422 8.972897 3.098868
## [5,] 3.129951 8.482944 3.181663
## [6,] 3.026166 7.566822 3.290681
```

References
- Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial Autoencoders. arXiv:1511.05644.

