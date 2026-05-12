#' SVM Classifier
#' @title Support Vector Machine Classification
#' @description Implements classification using support vector machines.
#' Wraps scikit-learn's `SVC` through `reticulate`.
#' @param attribute Target attribute name for model building.
#' @param slevels List of possible values for classification target.
#' @param C Regularization strength parameter.
#' @param kernel Kernel function type. One of `"rbf"`, `"linear"`, `"poly"`, or `"sigmoid"`.
#' @param gamma Kernel coefficient value. Use `"scale"`, `"auto"`, or a numeric value.
#' @param degree Polynomial degree when using `kernel = "poly"`.
#' @param coef0 Independent term value in polynomial and sigmoid kernels.
#' @param probability Whether to enable probability estimates.
#' @param class_weight Optional weights associated with classes.
#' @return A `skcla_svc` classifier object.
#'
#' @references
#' Cortes, C., & Vapnik, V. (1995). Support-Vector Networks.
#' @examples
#' \dontrun{
#' data(iris)
#' clf <- skcla_svc(
#'   attribute = "Species",
#'   slevels = levels(iris$Species),
#'   kernel = "rbf",
#'   C = 1
#' )
#' clf <- daltoolbox::fit(clf, iris)
#' pred <- predict(clf, iris)
#' table(pred, iris$Species)
#' }
#' @import daltoolbox
#' @export
skcla_svc <- function(attribute, slevels,
                      C = 1,
                      kernel = c("rbf", "linear", "poly", "sigmoid"),
                      gamma = "scale",
                      degree = 3,
                      coef0 = 0,
                      probability = FALSE,
                      class_weight = NULL) {
  kernel <- match.arg(kernel)

  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    attribute = attribute,
    slevels = slevels,
    C = as.numeric(C),
    kernel = kernel,
    gamma = gamma,
    degree = as.integer(degree),
    coef0 = as.numeric(coef0),
    probability = probability,
    class_weight = class_weight
  )

  obj <- c(obj, objex)
  class(obj) <- c("skcla_svc", cobj)
  obj
}

#' @import daltoolbox
#' @import reticulate
#' @exportS3Method fit skcla_svc
fit.skcla_svc <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_svc.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)

  if (is.null(obj$model)) {
    obj$model <- skcla_svc_create(
      C = obj$C,
      kernel = obj$kernel,
      gamma = obj$gamma,
      degree = obj$degree,
      coef0 = obj$coef0,
      probability = obj$probability,
      class_weight = obj$class_weight
    )
  }

  prepared <- prepare_skcla_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  obj$model <- skcla_svc_fit(obj$model, data, obj$attribute, obj$slevels)

  obj
}

#' @import daltoolbox
#' @import reticulate
#' @export
predict.skcla_svc <- function(object, x, ...) {
  if (!exists("skcla_svc_predict_proba")) {
    python_path <- system.file("python/skcla_svc.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }

  x <- prepare_skcla_predict_data(object, x)

  prediction <- skcla_svc_predict_proba(object$model, x)
  if (length(prediction) == 0) {
    prediction <- skcla_svc_predict(object$model, x)
  }
  prediction <- skcla_as_probability(prediction, object$slevels, object$model$classes_)

  prediction
}
