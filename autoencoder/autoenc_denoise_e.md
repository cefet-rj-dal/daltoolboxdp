## Autoencoder com Denoising (encode)

Este exemplo demonstra como usar um autoencoder com ruído (denoising) para aprender uma codificação robusta das janelas da série temporal. Durante o treino, ruído é adicionado à entrada.

Pré‑requisitos
- Python com PyTorch acessível via reticulate
- Pacotes R: daltoolbox, tspredit, daltoolboxdp, ggplot2
 
 Notas rápidas
 - Ideia: adicionar ruído na entrada e treinar o modelo para recuperar o sinal limpo.
 - Benefício: representações latentes mais robustas a perturbações.

``` r
# Denoising Autoencoder transformation (encode)

# Considering a dataset with $p$ numerical attributes. 

# The goal of the autoencoder is to reduce the dimension of $p$ to $k$, such that these $k$ attributes are enough to recompose the original $p$ attributes. 

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

auto <- autoenc_denoise_e(5, 3, num_epochs=1500)

auto <- fit(auto, train)
```


``` r
fit_loss <- data.frame(x=1:length(auto$train_loss), train_loss=auto$train_loss,val_loss=auto$val_loss)

grf <- plot_series(fit_loss, colors=c('Blue','Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_denoise_e/unnamed-chunk-7-1.png)
 

``` r
# A convergência deve ser estável; ruído em excesso pode dificultar o ajuste.
```


``` r
# Testando autoencoder
# Apresentando o conjunto de teste e exibindo codificação

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
##           [,1]       [,2]      [,3]
## [1,] 0.9756731 -0.5861361 0.7299481
## [2,] 0.9892359 -0.7131050 0.7412537
## [3,] 0.9668233 -0.8284920 0.7290662
## [4,] 0.9183574 -0.9070378 0.6973485
## [5,] 0.8438716 -0.9509255 0.6469927
## [6,] 0.7479969 -0.9574272 0.5811294
```

