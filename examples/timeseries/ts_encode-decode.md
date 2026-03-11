## Time Series Encoding and Reconstruction (encode-decode)

Time series windows of size p are encoded into k-dimensional latent vectors and then decoded back to p dimensions. Training minimizes reconstruction loss so that latent codes capture the essential structure of windows, enabling quality assessment via reconstruction error.

This example shows how to transform a time series into windows (p) and train an autoencoder to encode (p -> k) and reconstruct (k -> p) these windows, allowing evaluation of reconstruction quality.

Prerequisites
- R packages: daltoolbox, ggplot2
- Python with PyTorch accessible via reticulate (backend called internally)


``` r
# Loading required packages
library(daltoolbox)
```

Series for study


``` r
data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # inject a synthetic outlier for illustration
```


``` r
sw_size <- 5                         # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)        # series -> windows with p columns
ts_head(ts, 3)
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
```


``` r
library(ggplot2)
plot_ts(x = tsd$x, y = tsd$y) +
  theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-4](fig/ts_encode-decode/unnamed-chunk-4-1.png)

Data sampling


``` r
samp <- ts_sample(ts, test_size = 5)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```

Train the model (encode-decode)


``` r
auto <- autoenc_ed(5, 3)             # 5 -> 3 -> 5 dimensions
auto <- fit(auto, train)
```

Reconstruction evaluation (train)


``` r
print(head(train))                    # original windows (p columns)
```

```
##          t4        t3        t2        t1        t0
## 1 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## 2 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## 3 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
## 4 0.6816388 0.8414710 0.9489846 0.9974950 0.9839859
## 5 0.8414710 0.9489846 0.9974950 0.9839859 0.9092974
## 6 0.9489846 0.9974950 0.9839859 0.9092974 0.7780732
```

``` r
result <- transform(auto, train)      # reconstructed windows (p columns)
print(head(result))
```

```
##             [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] -0.00521685 0.2471274 0.4805990 0.6808790 0.8512622
## [2,]  0.25222406 0.4737628 0.6818197 0.8447692 0.9472243
## [3,]  0.48111042 0.6806357 0.8401164 0.9522001 0.9989684
## [4,]  0.67954576 0.8406144 0.9460032 0.9957585 0.9830227
## [5,]  0.84040332 0.9526547 0.9961526 0.9827881 0.9091274
## [6,]  0.94612324 0.9984223 0.9867116 0.9074917 0.7777395
```

Reconstruction of the test set


``` r
print(head(test))
```

```
##          t4        t3         t2         t1         t0
## 1 0.9893582 0.9226042  0.7984871  0.6247240  0.4121185
## 2 0.9226042 0.7984871  0.6247240  0.4121185  0.1738895
## 3 0.7984871 0.6247240  0.4121185  0.1738895 -0.4509067
## 4 0.6247240 0.4121185  0.1738895 -0.4509067 -0.3195192
## 5 0.4121185 0.1738895 -0.4509067 -0.3195192 -0.5440211
```

``` r
result <- transform(auto, test)
print(head(result))
```

```
##           [,1]       [,2]        [,3]         [,4]       [,5]
## [1,] 0.9910866 0.91773707  0.79943389  0.624025464  0.4083644
## [2,] 0.9296375 0.79586899  0.61920655  0.411642313  0.1733542
## [3,] 0.7939857 0.59869730  0.28673184  0.009827062 -0.2703653
## [4,] 0.5910214 0.36297092  0.08189377 -0.163898468 -0.4231845
## [5,] 0.2919053 0.05421017 -0.19777028 -0.439398766 -0.6743510
```

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)

