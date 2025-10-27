## Seleção de Atributos com Relief

Este exemplo usa o método Relief para estimar a relevância de atributos considerando vizinhos próximos e diferenças entre classes, ranqueando e selecionando atributos mais informativos para o alvo.

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
# Relief — passo a passo

# 1) Ajustar o seletor com alvo "Species"
myfeature <- fit(fs_relief("Species"), iris)

# 2) Ver os atributos selecionados
print(myfeature$features)
```

```
## [1] "Petal.Width"  "Petal.Length"
```

``` r
# 3) Transformar os dados para manter selecionados + alvo
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

