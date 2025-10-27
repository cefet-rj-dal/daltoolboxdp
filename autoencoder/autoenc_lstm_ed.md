## Autoencoder LSTM (encode-decode)

Este exemplo demonstra o uso de um Autoencoder baseado em LSTM para codificar janelas de série temporal (p -> k) e reconstruí‑las (k -> p). Assim, é possível avaliar a qualidade da reconstrução.

Pré‑requisitos
- Python com PyTorch acessível via reticulate
- Pacotes R: daltoolbox, tspredit, daltoolboxdp, ggplot2


``` r
# Instalando dependências do exemplo (se necessário)
install.packages("tspredit")
install.packages("daltoolboxdp")
```


``` r
# Carregando pacotes necessários
library(daltoolbox)
library(tspredit)
library(daltoolboxdp)
library(ggplot2)
```


``` r
# Conjunto de dados de exemplo (série -> janelas)
data(tsd)

sw_size <- 5                      # tamanho da janela deslizante (p)
ts <- ts_data(tsd$y, sw_size)     # converte série em janelas com p colunas

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
# Divisão em treino e teste
samp <- ts_sample(ts, test_size = 10)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```


``` r
# Criando o autoencoder LSTM (encode-decode): 5 -> 3 -> 5 dimensões
auto <- autoenc_lstm_ed(5, 3, num_epochs = 1500)

# Treinando o modelo
auto <- fit(auto, train)
```


``` r
# Curvas de aprendizado (perda de treino e validação por época)
fit_loss <- data.frame(
  x = 1:length(auto$train_loss),
  train_loss = auto$train_loss,
  val_loss = auto$val_loss
)
grf <- plot_series(fit_loss, colors = c('Blue', 'Orange'))
plot(grf)
```

![plot of chunk unnamed-chunk-7](fig/autoenc_lstm_ed/unnamed-chunk-7-1.png)


``` r
# Testando o autoencoder (reconstrução)
# Mostra amostras do conjunto de teste e a reconstrução (p colunas)
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
## [1,] 0.7976671
## [2,] 0.8587617
## [3,] 0.8917737
## [4,] 0.9054338
## [5,] 0.9049847
## [6,] 0.8927370
## 
## , , 2
## 
##           [,1]
## [1,] 0.9066344
## [2,] 0.9565270
## [3,] 0.9782443
## [4,] 0.9772234
## [5,] 0.9551523
## [6,] 0.9108632
## 
## , , 3
## 
##           [,1]
## [1,] 0.8929862
## [2,] 0.9308009
## [3,] 0.9444313
## [4,] 0.9374407
## [5,] 0.9096528
## [6,] 0.8581094
## 
## , , 4
## 
##           [,1]
## [1,] 0.9102679
## [2,] 0.9378524
## [3,] 0.9445603
## [4,] 0.9319301
## [5,] 0.8982355
## [6,] 0.8394521
## 
## , , 5
## 
##           [,1]
## [1,] 0.9106256
## [2,] 0.9355988
## [3,] 0.9395129
## [4,] 0.9232450
## [5,] 0.8841358
## [6,] 0.8170673
```


``` r
# Métricas de reconstrução por coluna: R² e MAPE
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
## [1] "t4 R2 teste: 0.7440820337429 MAPE: 0.0719812940780948"
## [1] "t3 R2 teste: 0.907830980259654 MAPE: 0.0662596681695922"
## [1] "t2 R2 teste: 0.9964201279753 MAPE: 0.0494247735683323"
## [1] "t1 R2 teste: 0.984646872253407 MAPE: 0.0631973823187948"
## [1] "t0 R2 teste: 0.949772742823873 MAPE: 0.163904768626404"
```

``` r
print(paste('Médias R2 teste:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Médias R2 teste: 0.916550551411027 MAPE: 0.0829535773522435"
```

