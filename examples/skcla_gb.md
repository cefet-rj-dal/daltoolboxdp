## Classificador Gradient Boosting — Visão Geral

Este exemplo utiliza o Gradient Boosting (scikit‑learn via reticulate) para classificar a base Iris.
Fluxo: dividir treino/teste, treinar, prever e avaliar (métricas de classificação).

Pré‑requisitos
- pacotes R: daltoolbox, daltoolboxdp
- Python acessível pelo reticulate (scikit‑learn instalado)


``` r
# Gradient Boosting Classifier

# Instalação (se necessário)

install.packages("daltoolboxdp")
```


``` r
# Carregando pacotes
library(daltoolbox)
library(daltoolboxdp)
```



``` r
# Carregando dataset Iris
iris <- datasets::iris
```


``` r
# Treino e avaliação com Gradient Boosting

slevels <- levels(iris$Species)                 # níveis da variável alvo

set.seed(1)
sr <- sample_random()                           # amostragem aleatória estratificada
sr <- train_test(sr, iris)                      # separa dados
iris_train <- sr$train
iris_test <- sr$test

# Codificação numérica do alvo para scikit‑learn (mantendo Species como alvo original)
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

# 1) Treinar
model <- skcla_gb("species_encoded", slevels)
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)

# 2) Avaliação no treino
iris_train_predictand <- adjust_class_label(iris_train[, "Species"])  # rótulos originais
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 39 81  0  0         1      1           1           1  1
```

``` r
# 3) Avaliação no teste
iris_test$species_encoded <- as.integer(as.factor(iris_test$Species))
iris_test_label <- iris_test[, !names(iris_test) %in% "Species"]
test_prediction <- predict(model, iris_test_label)

iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
test_eval <- evaluate(model, iris_test_predictand, test_prediction)
print(test_eval$metrics)
```

```
##    accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1 0.9333333 11 19  0  0         1      1           1           1  1
```
