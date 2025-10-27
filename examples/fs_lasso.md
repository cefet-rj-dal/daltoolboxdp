## Seleção de Atributos com Lasso

Este exemplo usa o Lasso (penalização L1) para selecionar atributos relevantes com base no relacionamento com a variável alvo. O L1 induz esparsidade nos coeficientes, removendo atributos menos importantes.

Pré‑requisitos
- pacotes R: daltoolbox, daltoolboxdp


``` r
# Instalação (se necessário)
#install.packages("daltoolboxdp")
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
# Lasso — passo a passo

# 1) Ajustar o seletor com alvo "Species"
myfeature <- fit(fs_lasso("Species"), iris)

# 2) Ver os atributos selecionados
print(myfeature$features)
```

```
## [1] "Sepal.Width"  "Petal.Length" "Petal.Width"
```

``` r
# 3) Transformar os dados para manter selecionados + alvo
data <- transform(myfeature, iris)
print(head(data))
```

```
##   Sepal.Width Petal.Length Petal.Width Species
## 1         3.5          1.4         0.2  setosa
## 2         3.0          1.4         0.2  setosa
## 3         3.2          1.3         0.2  setosa
## 4         3.1          1.5         0.2  setosa
## 5         3.6          1.4         0.2  setosa
## 6         3.9          1.7         0.4  setosa
```

