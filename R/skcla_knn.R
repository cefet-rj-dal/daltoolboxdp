#' K-Nearest Neighbors Classifier
#'@title K-Nearest Neighbors Classifier
#'@description Implements classification using the k-Nearest Neighbors algorithm.
#' Wraps scikit-learn's `KNeighborsClassifier` through `reticulate`.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param n_neighbors Number of neighbors to use for queries
#'@param weights Weight function used in prediction ('uniform', 'distance')
#'@param algorithm Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
#'@param leaf_size Leaf size passed to BallTree or KDTree
#'@param p Power parameter for the Minkowski metric
#'@param metric Distance metric for the tree ('euclidean', 'manhattan', 'chebyshev', 'minkowski', etc.)
#'@param metric_params Additional parameters for the metric function
#'@param n_jobs Number of parallel jobs for neighbor searches
#'@return A `skcla_knn` classifier object.
#'
#'@references
#' Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
#'@examples
#'\dontrun{
#'data(iris)
#'clf <- skcla_knn(attribute = 'Species', slevels = levels(iris$Species), n_neighbors = 7)
#'clf <- daltoolbox::fit(clf, iris)
#'pred <- predict(clf, iris)
#'table(pred, iris$Species)
#'}
#'
#'# More examples:
#'# https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/skcla_knn.md
#'@import daltoolbox
#'@export
skcla_knn <- function(attribute, slevels, 
                    n_neighbors = 5, 
                    weights = 'uniform', 
                    algorithm = 'auto', 
                    leaf_size = 30, 
                    p = 2, 
                    metric = 'minkowski', 
                    metric_params = NULL, 
                    n_jobs = NULL) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    n_neighbors = as.integer(n_neighbors),
    weights = weights,
    algorithm = algorithm,
    leaf_size = as.integer(leaf_size),
    p = as.integer(p),
    metric = metric,
    metric_params = metric_params,
    n_jobs = n_jobs
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("skcla_knn", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit skcla_knn
fit.skcla_knn <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_knn.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
  if (is.null(obj$model)) {
    obj$model <- skcla_knn_create(
      n_neighbors = obj$n_neighbors,
      weights = obj$weights,
      algorithm = obj$algorithm,
      leaf_size = obj$leaf_size,
      p = obj$p,
      metric = obj$metric,
      metric_params = obj$metric_params,
      n_jobs = obj$n_jobs
    )
  }
  
  data <- adjust_data.frame(data)
  obj$model <- skcla_knn_fit(obj$model, data, obj$attribute)
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.skcla_knn <- function(object, x, ...) {
  if (!exists("skcla_knn_predict")) {
    python_path <- system.file("python/skcla_knn.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  # Prepare features for prediction
  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]
  
  prediction <- skcla_knn_predict(object$model, x)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
