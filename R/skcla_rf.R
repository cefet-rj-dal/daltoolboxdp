#' Tree Ensemble
#' @title Random Forest Classifier
#' @description Implements classification using the Random Forest algorithm.
#' Wraps scikit-learn's `RandomForestClassifier` through `reticulate`.
#' @param attribute Target attribute name for model building.
#' @param slevels List of possible values for classification target.
#' @param n_estimators Number of trees in the forest.
#' @param max_depth Maximum tree depth value.
#' @param min_samples_split Minimum samples needed for an internal node split.
#' @param min_samples_leaf Minimum samples needed at a leaf node.
#' @param max_features Number of features to consider at each split. Use `"sqrt"`, `"log2"`, `NULL`, or a numeric value.
#' @param class_weight Optional weights associated with classes.
#' @param random_state Optional seed for reproducibility.
#' @return A `skcla_rf` classifier object.
#'
#' @references
#' Breiman, L. (2001). Random Forests. Machine Learning.
#' @examples
#' \dontrun{
#' data(iris)
#' clf <- skcla_rf(
#'   attribute = "Species",
#'   slevels = levels(iris$Species),
#'   n_estimators = 200,
#'   max_features = "sqrt"
#' )
#' clf <- daltoolbox::fit(clf, iris)
#' pred <- predict(clf, iris)
#' table(pred, iris$Species)
#' }
#' @import daltoolbox
#' @export
skcla_rf <- function(attribute, slevels,
                     n_estimators = 100,
                     max_depth = NULL,
                     min_samples_split = 2,
                     min_samples_leaf = 1,
                     max_features = "sqrt",
                     class_weight = NULL,
                     random_state = NULL) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    n_estimators = as.integer(n_estimators),
    max_depth = max_depth,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    max_features = max_features,
    class_weight = class_weight,
    random_state = if (!is.null(random_state)) as.integer(random_state) else NULL
  )

  obj <- c(obj, objex)
  class(obj) <- c("skcla_rf", cobj)
  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method fit skcla_rf
fit.skcla_rf <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_rf.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)

  if (is.null(obj$model)) {
    obj$model <- skcla_rf_create(
      n_estimators = obj$n_estimators,
      max_depth = obj$max_depth,
      min_samples_split = obj$min_samples_split,
      min_samples_leaf = obj$min_samples_leaf,
      max_features = obj$max_features,
      class_weight = obj$class_weight,
      random_state = obj$random_state
    )
  }

  data <- adjust_data.frame(data)
  obj$model <- skcla_rf_fit(obj$model, data, obj$attribute)

  obj
}

#' @import daltoolbox
#' @import reticulate
#' @export
predict.skcla_rf  <- function(object, x, ...) {
  if (!exists("skcla_rf_predict")) {
    python_path <- system.file("python/skcla_rf.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }

  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]

  prediction <- skcla_rf_predict(object$model, x)
  prediction <- adjust_class_label(prediction)

  prediction
}
