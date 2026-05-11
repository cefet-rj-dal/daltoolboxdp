#' K-Nearest Neighbors Classifier
#' @title K-Nearest Neighbors Classifier
#' @description Implements classification using the k-Nearest Neighbors algorithm.
#' Wraps scikit-learn's `KNeighborsClassifier` through `reticulate`.
#' @param attribute Target attribute name for model building.
#' @param slevels List of possible values for classification target.
#' @param n_neighbors Number of neighbors to use for queries.
#' @param weights Weight function used in prediction. One of `"uniform"` or `"distance"`.
#' @param metric Distance metric used by the neighbor search. One of
#'   `"euclidean"`, `"manhattan"`, `"chebyshev"`, or `"minkowski"`.
#' @return A `skcla_knn` classifier object.
#'
#' @references
#' Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
#' @examples
#' \dontrun{
#' data(iris)
#' clf <- skcla_knn(
#'   attribute = "Species",
#'   slevels = levels(iris$Species),
#'   n_neighbors = 7,
#'   weights = "distance"
#' )
#' clf <- daltoolbox::fit(clf, iris)
#' pred <- predict(clf, iris)
#' table(pred, iris$Species)
#' }
#' @import daltoolbox
#' @export
skcla_knn <- function(attribute, slevels,
                      n_neighbors = 5,
                      weights = c("uniform", "distance"),
                      metric = c("euclidean", "manhattan", "chebyshev", "minkowski")) {
  weights <- match.arg(weights)
  metric <- match.arg(metric)

  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    n_neighbors = as.integer(n_neighbors),
    weights = weights,
    metric = metric
  )

  obj <- c(obj, objex)
  class(obj) <- c("skcla_knn", cobj)
  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method fit skcla_knn
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
      metric = obj$metric
    )
  }

  data <- adjust_data.frame(data)
  obj$model <- skcla_knn_fit(obj$model, data, obj$attribute)

  obj
}

#' @import daltoolbox
#' @import reticulate
#' @export
predict.skcla_knn <- function(object, x, ...) {
  if (!exists("skcla_knn_predict")) {
    python_path <- system.file("python/skcla_knn.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }

  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]

  prediction <- skcla_knn_predict(object$model, x)
  prediction <- adjust_class_label(prediction)

  prediction
}
