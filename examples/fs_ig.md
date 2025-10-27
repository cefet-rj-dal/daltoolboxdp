## Seleção de Atributos com Information Gain (IG)

Este exemplo mostra como usar o método de Information Gain para ranquear atributos e selecionar um subconjunto relevante para o alvo. Em seguida, aplicamos a transformação para manter apenas os atributos escolhidos junto do alvo.

Pré‑requisitos
- pacotes R: daltoolbox, daltoolboxdp


``` r
# Instalação (se necessário)
install.packages("daltoolboxdp")
```


``` r
# Carregando pacotes
library(daltoolbox)
library(daltoolboxdp)
```



``` r
# Dados de exemplo
iris <- datasets::iris
```


``` r
# Information Gain (IG) — passo a passo

# 1) Ajustar o seletor de atributos (alvo: Species)
myfeature <- fit(fs_ig("Species"), iris)

# 2) Ver atributos selecionados
print(myfeature$features)
```

```
## [1] "Petal.Width"  "Petal.Length"
```

``` r
# 3) Aplicar transformação para manter apenas selecionados + alvo
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Petal.Width Petal.Length Species
## 1         0.2          1.4  setosa
## 2         0.2          1.4  setosa
## 3         0.2          1.3  setosa
## 4         0.2          1.5  setosa
## 5         0.2          1.4  setosa
## 6         0.4          1.7  setosa
```

