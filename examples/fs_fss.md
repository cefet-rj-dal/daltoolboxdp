## Seleção de Atributos com Forward Sequential Selection (FSS)

Este exemplo utiliza FSS (seleção sequencial forward) para construir um subconjunto de atributos adicionando, a cada passo, o atributo que mais melhora o critério de avaliação.

Pré‑requisitos
- pacotes R: daltoolbox, daltoolboxdp


``` r
# Feature Selection

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
# FSS — passo a passo

# 1) Ajustar o seletor com alvo "Species"
myfeature <- fit(fs_fss("Species"), iris)

# 2) Ver os atributos selecionados
print(myfeature$features)
```

```
## [1] "Sepal.Length" "Petal.Length" "Petal.Width"
```

``` r
# 3) Transformar os dados para manter selecionados + alvo
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Sepal.Length Petal.Length Petal.Width Species
## 1          5.1          1.4         0.2  setosa
## 2          4.9          1.4         0.2  setosa
## 3          4.7          1.3         0.2  setosa
## 4          4.6          1.5         0.2  setosa
## 5          5.0          1.4         0.2  setosa
## 6          5.4          1.7         0.4  setosa
```

