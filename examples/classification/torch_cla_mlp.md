## PyTorch Multi-Layer Perceptron (MLP) Classifier

This example uses the PyTorch-backed MLP classifier exposed by
`daltoolboxdp` to classify the Iris dataset. The workflow mirrors the
scikit-learn MLP example: split train/test, train, predict, and
evaluate.

Prerequisites - R packages: daltoolbox, daltoolboxdp - Python with
PyTorch accessible via reticulate

    # Installation (if needed)
    #install.packages("daltoolboxdp")

    library(daltoolbox)

    ## Warning: package 'daltoolbox' was built under R version 4.5.3

    ## 
    ## Attaching package: 'daltoolbox'

    ## The following object is masked from 'package:base':
    ## 
    ##     transform

    library(daltoolboxdp)

    # Loading Iris dataset
    iris <- datasets::iris

    # Training and evaluation with PyTorch MLP
    slevels <- levels(iris$Species)

    set.seed(1)
    sr <- sample_random()
    sr <- train_test(sr, iris)
    iris_train <- sr$train
    iris_test <- sr$test

    model <- torch_cla_mlp(
      attribute = "Species",
      slevels = slevels,
      input_size = 4L,
      hidden_sizes = c(16L, 8L),
      num_classes = 3L,
      epochs = 100L
    )

    model <- fit(model, iris_train)
    train_prediction <- predict(model, iris_train)

    iris_train_predictand <- adjust_class_label(iris_train[, "Species"])
    train_eval <- evaluate(model, iris_train_predictand, train_prediction)
    print(train_eval$metrics)

    ## NULL

    # Test prediction and evaluation
    test_prediction <- predict(model, iris_test)

    iris_test_predictand <- adjust_class_label(iris_test[, "Species"])
    test_eval <- evaluate(model, iris_test_predictand, test_prediction)
    print(test_eval$metrics)

    ## NULL

    # Predicted probabilities
    probabilities <- predict_proba.torch_cla_mlp(model, iris_test[, !names(iris_test) %in% "Species"])
    head(probabilities)

    ## [[1]]
    ## [1] 0.86886913 0.10290703 0.02822383
    ## 
    ## [[2]]
    ## [1] 0.92545813 0.06133946 0.01320234
    ## 
    ## [[3]]
    ## [1] 0.90158391 0.07967591 0.01874018
    ## 
    ## [[4]]
    ## [1] 0.85449731 0.11269601 0.03280667
    ## 
    ## [[5]]
    ## [1] 0.92619729 0.06137609 0.01242654
    ## 
    ## [[6]]
    ## [1] 0.956809938 0.037093319 0.006096772

Notes - By default, this example uses `validation_strategy = "static"`
and `stopping_rule = "none"`. - To enable early stopping, change
`stopping_rule` to `"patience"`, `"sma"`, `"ema"`, or `"h"`. - To switch
to the dynamic split, use `validation_strategy = "dynamic"`.

References - Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
Learning representations by back-propagating errors. - Paszke, A., et
al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning
Library.
