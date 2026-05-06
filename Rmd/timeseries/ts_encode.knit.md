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

![](C:/Git/dal/daltoolboxdp/examples/timeseries/doc/ts_encode_files/figure-docx/unnamed-chunk-4-1.png)<!-- -->

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
##             [,1]       [,2]      [,3]
## [1,] -0.58791876 -0.6623528 0.4329058
## [2,] -0.46181139 -0.8707486 0.5909081
## [3,] -0.29848197 -1.0075243 0.7167485
## [4,] -0.09607549 -1.0844626 0.7987470
## [5,]  0.09359595 -1.1052122 0.8367527
## [6,]  0.25734141 -1.0621089 0.8263159
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
##           [,1]       [,2]       [,3]
## [1,] 0.4616863 -0.8077081  0.6694039
## [2,] 0.4942211 -0.5852442  0.5281265
## [3,] 0.6992936 -0.1135772  0.3044598
## [4,] 0.5753309  0.2230003  0.1322699
## [5,] 0.3480665  0.6163263 -0.1515697
```

References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter on Autoencoders)
