## Codificação e Reconstrução de Séries (encode-decode)

Este exemplo mostra como transformar uma série temporal em janelas (p) e treinar um autoencoder para codificar (p → k) e reconstruir (k → p) essas janelas, permitindo avaliar a qualidade da reconstrução.

Pré‑requisitos
- pacotes R: daltoolbox, ggplot2
- Python com PyTorch acessível via reticulate (backend chamado internamente)


``` r
# Carregando pacotes necessários
library(daltoolbox)
```

## Série para estudo


``` r
data(tsd)
tsd$y[39] <- tsd$y[39] * 6   # injeta um outlier sintético para ilustração
```


``` r
sw_size <- 5                         # tamanho da janela deslizante (p)
ts <- ts_data(tsd$y, sw_size)        # série → janelas com p colunas
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

## Amostragem dos dados


``` r
samp <- ts_sample(ts, test_size = 5)
train <- as.data.frame(samp$train)
test  <- as.data.frame(samp$test)
```

## Treinando o modelo (encode-decode)


``` r
auto <- autoenc_ed(5, 3)             # 5 → 3 → 5 dimensões
auto <- fit(auto, train)
```

## Avaliação da reconstrução (treino)


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
result <- transform(auto, train)      # janelas reconstruídas (p colunas)
print(head(result))
```

```
##              [,1]      [,2]      [,3]      [,4]      [,5]
## [1,] -0.004245751 0.2486698 0.4804802 0.6832375 0.8490135
## [2,]  0.249563813 0.4737409 0.6837005 0.8419298 0.9492305
## [3,]  0.482917935 0.6841216 0.8420411 0.9526980 1.0006322
## [4,]  0.679013610 0.8401371 0.9437020 0.9972167 0.9843364
## [5,]  0.842465878 0.9516642 0.9954049 0.9838211 0.9105142
## [6,]  0.948686600 0.9983174 0.9868610 0.9075404 0.7791224
```

## Reconstrução do conjunto de teste


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
##           [,1]       [,2]        [,3]        [,4]       [,5]
## [1,] 0.9868255 0.92032683  0.80141675  0.62701482  0.4106435
## [2,] 0.9292169 0.80038536  0.62367135  0.41496181  0.1812990
## [3,] 0.8090640 0.60129547  0.27755004  0.02878652 -0.2629534
## [4,] 0.5980129 0.36838448  0.08149122 -0.16708124 -0.4286862
## [5,] 0.2953016 0.05227197 -0.20142742 -0.44397157 -0.6837102
```

