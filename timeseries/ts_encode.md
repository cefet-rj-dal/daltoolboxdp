## Codificação de Séries Temporais (encode)

Este exemplo mostra como transformar uma série temporal em janelas de tamanho fixo e treinar um autoencoder para aprender uma representação latente compacta (p → k) dessas janelas.

Pré‑requisitos
- pacotes R: daltoolbox, ggplot2
- Python com PyTorch acessível via reticulate (o backend é carregado por funções internas)


``` r
# Carregando pacotes necessários
library(daltoolbox)
```

## Série para estudo


``` r
data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # injeta um outlier sintético para ilustração no gráfico
```


``` r
sw_size <- 5                         # tamanho da janela deslizante (p)
ts <- ts_data(tsd$y, sw_size)        # converte a série em janelas com p colunas
ts_head(ts, 3)                       # visualiza as 3 primeiras janelas
```

```
##             t4        t3        t2        t1        t0
## [1,] 0.0000000 0.2474040 0.4794255 0.6816388 0.8414710
## [2,] 0.2474040 0.4794255 0.6816388 0.8414710 0.9489846
## [3,] 0.4794255 0.6816388 0.8414710 0.9489846 0.9974950
```


``` r
library(ggplot2)
plot_ts(x = tsd$x, y = tsd$y) +      # gráfico da série com o ponto atípico marcado pelo pico
  theme(text = element_text(size = 16))
```

![plot of chunk unnamed-chunk-4](fig/ts_encode/unnamed-chunk-4-1.png)

## Amostragem dos dados


``` r
samp <- ts_sample(ts, test_size = 5) # separa as últimas 5 janelas para teste
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```

## Treinando o modelo


``` r
auto <- autoenc_e(5, 3)              # reduz de 5 → 3 dimensões (p → k)
auto <- fit(auto, train)
```

## Avaliação da codificação (treino)


``` r
print(head(train))                    # janelas originais (p colunas)
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
result <- transform(auto, train)      # codificações (k colunas)
print(head(result))
```

```
##           [,1]        [,2]     [,3]
## [1,] 0.6414449 -0.36973172 1.129125
## [2,] 0.8229654 -0.06447159 1.355677
## [3,] 0.9272192  0.25786611 1.499546
## [4,] 0.9709718  0.55090469 1.567538
## [5,] 0.9543616  0.79748935 1.555104
## [6,] 0.8768654  0.98048949 1.460820
```

## Codificação do conjunto de teste


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
##            [,1]      [,2]      [,3]
## [1,]  0.5950999 1.0993470 1.1122403
## [2,]  0.3932760 1.0232307 0.8621345
## [3,]  0.1322747 1.0763371 0.5148003
## [4,] -0.1042039 0.6923359 0.1891492
## [5,] -0.3764235 0.4324733 0.0455029
```

