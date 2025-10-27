## Autoencoder Adversarial (encode-decode)

Este exemplo mostra como treinar e usar um Autoencoder Adversarial (AAE) no modo encode-decode: o modelo comprime janelas de p para k dimensões e, em seguida, reconstrói de volta para p, permitindo avaliar o erro de reconstrução.

Pré‑requisitos
- Python com PyTorch acessível via reticulate
- Pacotes R: daltoolbox, tspredit, daltoolboxdp, ggplot2

Notas rápidas
- Avaliação: qualidade de reconstrução medida, por exemplo, via R² e MAPE por coluna da janela.
- Hiperparâmetros: `num_epochs`, `batch_size` influenciam convergência e estabilidade adversarial.


``` r
# Instalando dependências do exemplo (se necessário)
#install.packages("tspredit")
#install.packages("daltoolboxdp")
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
# Criando o autoencoder adversarial (encode-decode): 5 -> 3 -> 5 dimensões
auto <- autoenc_adv_ed(5, 3, batch_size = 3, num_epochs = 1500)

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

![plot of chunk unnamed-chunk-7](fig/autoenc_adv_ed/unnamed-chunk-7-1.png)


``` r
# Testando o autoencoder (reconstrução)
# Mostra amostras do conjunto de teste e a reconstrução gerada (p colunas)
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
##           [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] 0.8515993 0.9055303 0.9264832 0.9258715 0.8865762
## [2,] 0.8792633 0.9296120 0.9473595 0.9474165 0.9116967
## [3,] 0.8907688 0.9388611 0.9552834 0.9553570 0.9217457
## [4,] 0.8889199 0.9374500 0.9540140 0.9540896 0.9201584
## [5,] 0.8720955 0.9236389 0.9420888 0.9419913 0.9052253
## [6,] 0.8380271 0.8929982 0.9148843 0.9138092 0.8737197
```


``` r
# Métricas de reconstrução por coluna: R² e MAPE
# Observação: MAPE pode ser sensível a valores próximos de zero.
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
## [1] "t4 R2 teste: 0.330045151311239 MAPE: 0.171485470747373"
## [1] "t3 R2 teste: 0.88838766928187 MAPE: 0.0962939665543128"
## [1] "t2 R2 teste: 0.960612560838776 MAPE: 0.0429462170623562"
## [1] "t1 R2 teste: 0.910201756900088 MAPE: 0.132826902133761"
## [1] "t0 R2 teste: 0.850324108959418 MAPE: 0.318265658329421"
```

``` r
print(paste('Médias R2 teste:', mean(r2), 'MAPE:', mean(mape)))
```

```
## [1] "Médias R2 teste: 0.787914249458278 MAPE: 0.152363642965445"
```

