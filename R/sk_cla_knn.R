#'@title K-Nearest Neighbors Classifier
#'@description Classifies using the K-Nearest Neighbors (KNN) algorithm.
#' It wraps the sklearn library.
#'@param attribute attribute target to model building
#'@param slevels Possible values for the target classification
#'@param n_neighbors number of neighbors to use for classification
#'@param weights weight function used in prediction ('uniform', 'distance')
#'@param algorithm algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
#'@param leaf_size leaf size passed to the tree algorithms
#'@param p power parameter for Minkowski metric
#'@param metric distance metric to use ('minkowski', 'euclidean', 'manhattan', etc.)
#'@return A KNN classifier object
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_knn("Species", slevels, k=3)
#'
#'# preparing dataset for random sampling
#'sr <- sample_random()
#'sr <- train_test(sr, iris)
#'train <- sr$train
#'test <- sr$test
#'
#'model <- fit(model, train)
#'
#'prediction <- predict(model, test)
#'predictand <- adjust_class_label(test[,"Species"])
#'test_eval <- evaluate(model, predictand, prediction)
#'test_eval$metrics
#'@export
cla_knn <- function(attribute, slevels, n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski') {
  obj <- list(attribute = attribute, slevels = slevels, n_neighbors = n_neighbors, weights = weights,
              algorithm = algorithm, leaf_size = leaf_size, p = p, metric = metric)

  class(obj) <- c("cla_knn", class(obj))
  return(obj)
}


#'@import reticulate
#'@export
fit.cla_knn <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("cla_knn_create"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_knn.py")

  # Check if the model is already initialized, otherwise create it
  if (is.null(obj$model)) {
    obj$model <- cla_knn_create(
      as.integer(obj$n_neighbors),
      obj$weights,
      obj$algorithm,
      as.integer(obj$leaf_size),
      as.integer(obj$p),
      obj$metric
      )
  }

  # Adjust the data frame
  data <- adjust_data.frame(data)

  # Fit the model using the KNN function and the attributes from obj
  obj$model <- cla_knn_fit(obj$model, data, obj$attribute)
  return(obj)
}

#'@import reticulate
#'@export
predict.cla_knn  <- function(obj, data, ...) {
  if (!exists("cla_knn_predict"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_knn.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- cla_knn_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
