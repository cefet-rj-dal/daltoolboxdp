#'@title KNN
#'@description Creates a time series prediction object that uses the KNN.
#' It wraps the sklearn library.
#'@param preprocess normalization
#'@param n_neighbors number of neighbors for KNN
#'@return a `ts_knn` object.
#'@examples
#'#Use the same example of ts_mlp changing the constructor to:
#'model <- ts_knn(ts_norm_gminmax(), n_neighbors=5)
#'@import reticulate
#'@export
ts_knn <- function(preprocess = NA, n_neighbors = 5) {
  obj <- ts_regsw(preprocess, n_neighbors)
  obj$n_neighbors <- n_neighbors
  class(obj) <- append("ts_knn", class(obj))

  return(obj)
}

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
