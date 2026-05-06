About the method - `torch_reg_mlp`: PyTorch-backed multilayer perceptron
for regression. - Hyperparameters: `hidden_sizes` (hidden-layer
width/depth), `dropout`, `epochs`, `lr`, and the unified
validation/early-stopping controls.

Didactic goal: read this example as the same numeric-prediction workflow
used by `reg_mlp`. The Experiment Line stays the same; the practical
change is the constructor.

Environment setup.

    # Regression MLP with PyTorch

    # installation
    #install.packages("daltoolboxdp")

    # loading DAL
    library(daltoolbox)

    ## Warning: package 'daltoolbox' was built under R version 4.5.3

    ## 
    ## Attaching package: 'daltoolbox'

    ## The following object is masked from 'package:base':
    ## 
    ##     transform

    library(daltoolboxdp)

Load Boston dataset (MASS) and inspect types.

    # Dataset for regression analysis

    library(MASS)
    data(Boston)
    print(t(sapply(Boston, class)))

    ##      crim      zn        indus     chas      nox       rm        age      
    ## [1,] "numeric" "numeric" "numeric" "integer" "numeric" "numeric" "numeric"
    ##      dis       rad       tax       ptratio   black     lstat     medv     
    ## [1,] "numeric" "integer" "numeric" "numeric" "numeric" "numeric" "numeric"

    head(Boston)

    ##      crim zn indus chas   nox    rm  age    dis rad tax ptratio  black lstat
    ## 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296    15.3 396.90  4.98
    ## 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242    17.8 396.90  9.14
    ## 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242    17.8 392.83  4.03
    ## 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222    18.7 394.63  2.94
    ## 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222    18.7 396.90  5.33
    ## 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222    18.7 394.12  5.21
    ##   medv
    ## 1 24.0
    ## 2 21.6
    ## 3 34.7
    ## 4 33.4
    ## 5 36.2
    ## 6 28.7

Optional conversion to matrix.

    # for performance, you can convert to matrix
    Boston <- as.matrix(Boston)

Random and reproducible train/test split.

    # preparing dataset for random sampling
    set.seed(1)
    sr <- sample_random()
    sr <- train_test(sr, Boston)
    boston_train <- sr$train
    boston_test <- sr$test

Train MLP: define the hidden architecture and training controls.

    # Training

    model <- torch_reg_mlp(
      attribute = "medv",
      input_size = ncol(Boston) - 1L,
      hidden_sizes = c(16L, 8L),
      epochs = 100L
    )
    model <- fit(model, boston_train)

Training evaluation.

    # Model adjustment

    train_prediction <- predict(model, boston_train)
    boston_train_predictand <- boston_train[, "medv"]
    train_eval <- evaluate(model, boston_train_predictand, train_prediction)
    print(train_eval$metrics)

    ##        mse     smape        R2
    ## 1 46.05783 0.2042328 0.4882932

Test evaluation.

    # Test

    test_prediction <- predict(model, boston_test)
    boston_test_predictand <- boston_test[, "medv"]
    test_eval <- evaluate(model, boston_test_predictand, test_prediction)
    print(test_eval$metrics)

    ##        mse     smape       R2
    ## 1 31.19235 0.2139897 0.481643

Notes - Default configuration uses `validation_strategy = "static"` and
`stopping_rule = "none"`. - To activate early stopping, set
`stopping_rule` to `"patience"`, `"sma"`, `"ema"`, or `"h"`. - To
activate dynamic validation splits, use
`validation_strategy = "dynamic"`.

References - Bishop, C. M. (1995). Neural Networks for Pattern
Recognition. Oxford University Press. - Paszke, A., et al. (2019).
PyTorch: An Imperative Style, High-Performance Deep Learning Library.
