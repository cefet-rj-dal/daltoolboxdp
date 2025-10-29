## Time Series Encoding (encode)

We use a sliding-window embedding to convert a univariate series into fixed-length vectors of length p. A feed-forward autoencoder is trained to minimize reconstruction error, and its bottleneck (k < p) provides a compact encoding that preserves salient information for downstream tasks.

This example shows how to transform a time series into fixed-size windows and train an autoencoder to learn a compact latent representation (p -> k) of these windows.

Prerequisites
- R packages: daltoolbox, ggplot2
- Python with PyTorch accessible via reticulate (the backend is loaded by internal functions)


``` r
# Loading required packages
library(daltoolbox)
```

Series for study


``` r
data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # inject a synthetic outlier for illustration in the plot
```


``` r
sw_size <- 5                         # sliding window size (p)
ts <- ts_data(tsd$y, sw_size)        # convert the series into windows with p columns
ts_head(ts, 3)                       # view the first 3 windows
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
```


``` r
library(ggplot2)
plot_ts(x = tsd$x, y = tsd$y) +      # series plot with the outlier peak
  theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-4](fig/ts_encode/unnamed-chunk-4-1.png)

Data sampling


``` r
samp <- ts_sample(ts, test_size = 5) # hold out the last 5 windows for test
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```

Train the model


``` r
auto <- autoenc_e(5, 3)              # reduce from 5 -> 3 dimensions (p -> k)
auto <- fit(auto, train)
```

Encoding evaluation (train)


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
result <- transform(auto, train)      # encodings (k columns)
print(head(result))
```

```
##           [,1]       [,2]      [,3]
## [1,] 0.2795950 -0.7417191 -1.022133
## [2,] 0.6224524 -0.9036327 -1.254119
## [3,] 0.9344380 -1.0103003 -1.430671
## [4,] 1.1814177 -1.0467048 -1.525883
## [5,] 1.3597012 -1.0049314 -1.531249
## [6,] 1.4649724 -0.9028794 -1.456295
```

Encoding of the test set


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
##           [,1]        [,2]        [,3]
## [1,] 1.4472004 -0.57537431 -1.12094009
## [2,] 1.3211761 -0.35078946 -0.85330904
## [3,] 1.1107997  0.02935538 -0.31218180
## [4,] 0.8174326  0.24761176  0.02247674
## [5,] 0.4723179  0.37613583  0.39211157
```

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
