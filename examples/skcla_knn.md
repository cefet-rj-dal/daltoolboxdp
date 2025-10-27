## Classificador K-Nearest Neighbors (KNN)

Este exemplo utiliza KNN (scikit‑learn via reticulate) para classificar a base Iris. Fluxo: dividir treino/teste, treinar, prever e avaliar.

Pré‑requisitos
- pacotes R: daltoolbox, daltoolboxdp
- Python acessível pelo reticulate (scikit‑learn instalado)


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
# Carregando dataset Iris
iris <- datasets::iris
```


``` r
# Treino e avaliação com KNN

slevels <- levels(iris$Species)

set.seed(1)
sr <- sample_random()
sr <- train_test(sr, iris)
iris_train <- sr$train
iris_test <- sr$test

# Codificação numérica do alvo para scikit‑learn
iris_train$species_encoded <- as.integer(as.factor(iris_train$Species))
iris_train_label <- iris_train[, !names(iris_train) %in% "Species"]

model <- skcla_knn("species_encoded", slevels, n_neighbors = 1)
model <- fit(model, iris_train_label)
train_prediction <- predict(model, iris_train_label)

iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
train_eval <- evaluate(model, iris_train_predictand, train_prediction)
print(train_eval$metrics)
```

```
##   accuracy TP TN FP FN precision recall sensitivity specificity f1
## 1        1 39 81  0  0         1      1           1           1  1
```

``` r
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
