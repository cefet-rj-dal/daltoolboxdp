#' Tree Boosting
#' @title Gradient Boosting Classifier
#' @description Implements a classifier using the Gradient Boosting algorithm.
#' Wraps scikit-learn's `GradientBoostingClassifier` through `reticulate`.
#' @param attribute Target attribute name for model building.
#' @param slevels Possible values for the target classification.
#' @param n_estimators Number of boosting stages to perform.
#' @param learning_rate Learning rate that shrinks the contribution of each tree.
#' @param max_depth Maximum depth of the individual regression estimators.
#' @param subsample Fraction of samples used to fit each stage.
#' @param min_samples_split Minimum number of samples required to split an internal node.
#' @param min_samples_leaf Minimum number of samples required to be at a leaf node.
#' @param loss Loss function to be optimized. One of `"log_loss"` or `"exponential"`.
#' @param random_state Optional random number generator seed.
#' @return A `skcla_gb` classifier object.
#'
#' @references
#' Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
#' @examples
#' \dontrun{
#' data(iris)
#' clf <- skcla_gb(
#'   attribute = "Species",
#'   slevels = levels(iris$Species),
#'   n_estimators = 150,
#'   learning_rate = 0.05
#' )
#' clf <- daltoolbox::fit(clf, iris)
#' pred <- predict(clf, iris)
#' table(pred, iris$Species)
#' }
#' @import daltoolbox
#' @export
skcla_gb <- function(attribute, slevels,
                     n_estimators = 100,
                     learning_rate = 0.1,
                     max_depth = 3,
                     subsample = 1,
                     min_samples_split = 2,
                     min_samples_leaf = 1,
                     loss = c("log_loss", "exponential"),
                     random_state = NULL) {
  loss <- match.arg(loss)

  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    n_estimators = as.integer(n_estimators),
    learning_rate = as.numeric(learning_rate),
    max_depth = as.integer(max_depth),
    subsample = as.numeric(subsample),
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    loss = loss,
    random_state = if (!is.null(random_state)) as.integer(random_state) else NULL
  )
  obj <- c(obj, objex)
  class(obj) <- c("skcla_gb", cobj)
  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method fit skcla_gb
fit.skcla_gb <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_gb.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)

  if (is.null(obj$model)) {
    obj$model <- skcla_gb_create(
      n_estimators = obj$n_estimators,
      learning_rate = obj$learning_rate,
      max_depth = obj$max_depth,
      subsample = obj$subsample,
      min_samples_split = obj$min_samples_split,
      min_samples_leaf = obj$min_samples_leaf,
      loss = obj$loss,
      random_state = obj$random_state
    )
  }

  data <- adjust_data.frame(data)
  obj$model <- skcla_gb_fit(obj$model, data, obj$attribute, obj$slevels)

  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method predict skcla_gb
predict.skcla_gb  <- function(object, x, ...) {
  if (!exists("skcla_gb_predict")) {
    python_path <- system.file("python/skcla_gb.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }

  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]

  prediction <- skcla_gb_predict(object$model, x)
  prediction <- adjust_class_label(prediction)

  prediction
}
