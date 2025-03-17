#'@title Time Series K-Nearest Neighbors
#'@description Creates a time series prediction object using KNN.
#' It wraps the sklearn library.
#'@param preprocess preprocessing method for the time series data
#'@param n_neighbors number of neighbors to use for prediction
#'@param weights weight function used in prediction ('uniform', 'distance')
#'@param algorithm algorithm used to compute nearest neighbors
#'@param leaf_size leaf size passed to BallTree or KDTree
#'@param p power parameter for the Minkowski metric
#'@param metric distance metric for the tree
#'@return A time series KNN object
#'@examples
#'# Example code:
#'model <- ts_knn(ts_norm_gminmax(), n_neighbors=5)
#'# Add fitting and prediction examples
#'@import reticulate
#'@export
ts_knn <- function(preprocess = NA, n_neighbors = 5) {
  obj <- ts_regsw(preprocess, n_neighbors)
  obj$n_neighbors <- n_neighbors
  class(obj) <- append("ts_knn", class(obj))

  return(obj)
}

#'@method do_fit ts_knn
#'@export
do_fit.ts_knn <- function(obj, x, y) {
  if (!exists("knn_create"))
    reticulate::source_python(system.file("python", "inst/python/sklearn/ts_knn.py", package = "daltoolbox"))

  if (is.null(obj$model))
    obj$model <- knn_create(obj$n_neighbors)

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- knn_fit(obj$model, df_train)

  return(obj)
}

#'@method do_predict ts_knn
#'@export
do_predict.ts_knn <- function(obj, x) {
  if (!exists("knn_predict"))
    reticulate::source_python(system.file("python", "inst/python/sklearn/ts_knn.py", package = "daltoolbox"))

  X_values <- as.data.frame(x)
  X_values$t0 <- 0
  prediction <- knn_ts_predict(obj$model, X_values)
  prediction <- as.vector(prediction)
  return(prediction)
}
