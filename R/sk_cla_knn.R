#' K-Nearest Neighbors Classifier
#'@title K-Nearest Neighbors Classifier
#'@description Implements classification using the K-Nearest Neighbors algorithm.
#' This function wraps the KNeighborsClassifier from Python's scikit-learn library.
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
#'@return A K-Nearest Neighbors classifier object
#'@return `cla_knn` object
#'@examples
#'#See an example of using `cla_knn` at this
#'#https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/cla_knn.md
#'@import daltoolbox
#'@export
cla_knn <- function(attribute, slevels, 
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
  class(obj) <- c("cla_knn", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit cla_knn
fit.cla_knn <- function(obj, data, ...) {
  python_path <- system.file("python/sklearn/cla_knn.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
  if (is.null(obj$model)) {
    obj$model <- knn_create(
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
  obj$model <- knn_fit(obj$model, data, obj$attribute)
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.cla_knn <- function(object, x, ...) {
  if (!exists("knn_predict")) {
    python_path <- system.file("python/sklearn/cla_knn.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]
  
  prediction <- knn_predict(object$model, x)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
