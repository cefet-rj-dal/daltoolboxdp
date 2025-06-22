
``` r
# LSTM Autoencoder transformation (encode-decode)

# Considering a dataset with $p$ numerical attributes. 
 
# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensionals the data is returned back to $p$ dimensions. The higher the quality of autoencoder the similiar is the output from the input. 

# installing packages

install.packages("tspredit")
install.packages("daltoolboxdp")
```


``` r
# loading DAL
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```


``` r
# dataset for example 

data(tsd)

sw_size <- 5
ts <- ts_data(tsd$y, sw_size)

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
# applying data normalization

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
# spliting into training and test

samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```


``` r
# creating autoencoder - reduce from 5 to 3 dimensions

auto <- autoenc_lstm_ed(5, 3, num_epochs=1500)

auto <- fit(auto, train)
```


``` r
# learning curves

fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_lstm_ed/unnamed-chunk-7-1.png)


``` r
# testing autoencoder
# presenting the original test set and display encoding

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
## , , 1
## 
##           [,1]
## [1,] 0.7927169
## [2,] 0.8538489
## [3,] 0.8871050
## [4,] 0.9010271
## [5,] 0.9007320
## [6,] 0.8883963
## 
## , , 2
## 
##           [,1]
## [1,] 0.9079087
## [2,] 0.9587334
## [3,] 0.9815943
## [4,] 0.9819568
## [5,] 0.9615524
## [6,] 0.9190549
## 
## , , 3
## 
##           [,1]
## [1,] 0.8916554
## [2,] 0.9302931
## [3,] 0.9451305
## [4,] 0.9398216
## [5,] 0.9143277
## [6,] 0.8655986
## 
## , , 4
## 
##           [,1]
## [1,] 0.9101498
## [2,] 0.9386697
## [3,] 0.9467285
## [4,] 0.9360130
## [5,] 0.9049715
## [6,] 0.8494796
## 
## , , 5
## 
##           [,1]
## [1,] 0.9104216
## [2,] 0.9365193
## [3,] 0.9420010
## [4,] 0.9279501
## [5,] 0.8919412
## [6,] 0.8287690
```


``` r
result <- as.data.frame(result)
names(result) <- names(test)
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
## [1] "t4 R2 test: 0.743986820515025 MAPE: 0.0751530030551026"
## [1] "t3 R2 test: 0.915907164999694 MAPE: 0.0605331331085126"
## [1] "t2 R2 test: 0.996218151243702 MAPE: 0.0416127637933464"
## [1] "t1 R2 test: 0.979718059108326 MAPE: 0.0746331817846983"
## [1] "t0 R2 test: 0.941304264062905 MAPE: 0.18351817004376"
```

``` r
print(paste('Means R2 test:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Means R2 test: 0.915426891985931 MAPE: 0.087090050357084"
```

