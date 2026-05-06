## PyTorch MLP Classifier

This example shows how to use the PyTorch-backed MLP classifier exposed by `daltoolboxdp`. The workflow stays close to the `daltoolbox` classification line: prepare data, split into train/test, fit the model, inspect predictions, and optionally inspect class probabilities.

Prerequisites
- R packages: daltoolbox, daltoolboxdp
- Python with PyTorch accessible via reticulate


``` r
# Installation (if needed)
#install.packages("daltoolboxdp")
```


``` r
library(daltoolbox)
library(daltoolboxdp)
```


``` r
# Prepare Iris with a numeric target for the current wrapper
iris_torch <- data.frame(
  iris[, 1:4],
  species_encoded = as.integer(iris$Species)
)

set.seed(1)
idx <- sample(seq_len(nrow(iris_torch)), size = floor(0.8 * nrow(iris_torch)))
iris_train <- iris_torch[idx, ]
iris_test <- iris_torch[-idx, ]
```


``` r
# Fit the classifier
model <- torch_cla_mlp(
  attribute = "species_encoded",
  slevels = c(1L, 2L, 3L),
  input_size = 4L,
  hidden_sizes = c(16L, 8L),
  num_classes = 3L,
  epochs = 100L
)

model <- fit(model, iris_train)
```


``` r
# Predicted classes
prediction <- predict(model, iris_test)
head(prediction)
```

```
##      1 2 3
## [1,] 1 0 0
## [2,] 1 0 0
## [3,] 1 0 0
## [4,] 1 0 0
## [5,] 1 0 0
## [6,] 1 0 0
```


``` r
# Predicted probabilities
probabilities <- predict_proba.torch_cla_mlp(model, iris_test[, 1:4])
str(probabilities[1:3])
```

```
## List of 3
##  $ : num [1:3] 0.90684 0.08488 0.00828
##  $ : num [1:3] 0.94277 0.05336 0.00387
##  $ : num [1:3] 0.93306 0.06212 0.00482
```

Notes
- By default, this example uses `validation_strategy = "static"` and `stopping_rule = "none"`.
- To enable early stopping, change `stopping_rule` to `"patience"`, `"sma"`, `"ema"`, or `"h"`.
- To switch to the dynamic split, use `validation_strategy = "dynamic"`.

References
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
