## Autoencoder Convolucional (encode-decode)

Este exemplo demonstra como usar um autoencoder convolucional 1D para codificar e reconstruir janelas de uma série temporal. Após reduzir de p para k dimensões, o modelo reconstrói de volta para p, permitindo avaliar o erro de reconstrução.

Pré‑requisitos
- Python com PyTorch acessível via reticulate
- Pacotes R: daltoolbox, tspredit, daltoolboxdp, ggplot2
 
 Notas rápidas
 - Reconstrução: compare entrada e saída para verificar se os padrões locais foram preservados.
 - Métricas: R² e MAPE por coluna ajudam a medir a qualidade por passo da janela.

``` r
# Convolutional Autoencoder transformation (encode-decode)

# Considering a dataset with $p$ numerical attributes. 

# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. However from the $k$ dimensionals the data is returned back to $p$ dimensions. The higher the quality of autoencoder the similiar is the output from the input. 

# installing packages

install.packages("tspredit")
install.packages("daltoolboxdp")
```


``` r
# Carregando pacotes
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```


``` r
# Dataset de exemplo (série -> janelas) 

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
# Normalização (min-max por grupo)

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
# Divisão treino/teste

samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test <- as.data.frame(samp$test)
```


``` r
# Treinando autoencoder (reduz 5 -> 3)

auto <- autoenc_conv_ed(5, 3)

auto <- fit(auto, train)
```


``` r
fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_conv_ed/unnamed-chunk-7-1.png)


``` r
# Testando autoencoder
# Apresentando o conjunto de teste e exibindo reconstrução

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
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.7350522 0.8451846 0.9175721 0.9551076 0.9580944
## [2,] 0.8408093 0.9095408 0.9467298 0.9662982 0.9594948
## [3,] 0.9086373 0.9395595 0.9554131 0.9637793 0.9463183
## [4,] 0.9442120 0.9525085 0.9533068 0.9487289 0.9136994
## [5,] 0.9598675 0.9535236 0.9376767 0.9079835 0.8431801
## [6,] 0.9625239 0.9414502 0.8985656 0.8223424 0.7194853
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
  print(paste(col, 'R2 teste:', r2_col, 'MAPE:', mape_col))
}
```

```
## [1] "t4 R2 teste: 0.983630142182545 MAPE: 0.0166417209415435"
## [1] "t3 R2 teste: 0.973015334778284 MAPE: 0.0223837161040003"
## [1] "t2 R2 teste: 0.990854085601842 MAPE: 0.0189836249166383"
## [1] "t1 R2 teste: 0.997112470874845 MAPE: 0.0148726794853899"
## [1] "t0 R2 teste: 0.9944221154644 MAPE: 0.0193923005139626"
```

``` r
print(paste('Médias R2 teste:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Médias R2 teste: 0.987806829780383 MAPE: 0.0184548083923069"
```
 

``` r
# Observação: cuidado com divisões por valores muito próximos de zero ao calcular o MAPE.
```

