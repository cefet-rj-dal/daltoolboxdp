#' Naive Bayes Classifier
#'@title Gaussian Naive Bayes Classifier
#'@description Implements classification using Gaussian Naive Bayes.
#' Wraps scikit-learn's `GaussianNB` through `reticulate`.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param var_smoothing Portion of the largest variance of all features that is added to variances
#'@return A `skcla_nb` classifier object.
#'
#'@references
#' Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. (Gaussian Naive Bayes)
#'@examples
#'\dontrun{
#'data(iris)
#'
#'# Gaussian Naive Bayes for multi-class iris
#'clf <- skcla_nb(attribute = 'Species', slevels = levels(iris$Species))
#'clf <- daltoolbox::fit(clf, iris)
#'pred <- predict(clf, iris)
#'table(pred, iris$Species)
#'}
#'
#'# More examples:
#'# https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/skcla_nb.md
#'@import daltoolbox
#'@export
skcla_nb <- function(attribute, slevels, var_smoothing=1e-9) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    var_smoothing = as.numeric(var_smoothing)
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("skcla_nb", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit skcla_nb
fit.skcla_nb <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_nb.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
  if (any(is.na(data))) {
    warning("Missing values detected in the data. These will be handled as part of the process.")
  }
  
  if (is.null(obj$model)) {
    obj$model <- skcla_nb_create(
      var_smoothing = obj$var_smoothing
    )
    
    if (is.null(obj$model)) {
      stop("Failed to create Naive Bayes model.")
    }
  }
  
  prepared <- prepare_skcla_fit(obj, data)
  obj <- prepared$obj
  data <- prepared$data
  
  if (!obj$attribute %in% names(data)) {
    stop(paste("Attribute", obj$attribute, "not found in the data."))
  }
  
  #message("Fitting model with data dimensions: ", nrow(data), " x ", ncol(data))
  #message("Target attribute: ", obj$attribute)
  
  obj$model <- skcla_nb_fit(obj$model, data, obj$attribute)
  
  if (is.null(obj$model)) {
    stop("Failed to fit Naive Bayes model.")
  }
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.skcla_nb <- function(object, x, ...) {
  if (!exists("skcla_nb_predict")) {
    python_path <- system.file("python/skcla_nb.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  if (any(is.na(x))) {
    warning("Missing values detected in the prediction data. These will be handled as part of the process.")
  }
  
  # Prepare features for prediction
  x <- prepare_skcla_predict_data(object, x)
  
  #message("Predicting with data dimensions: ", nrow(x), " x ", ncol(x))
  
  prediction <- skcla_nb_predict_proba(object$model, x)
  
  if (is.null(prediction) || length(prediction) == 0) {
    warning("Prediction returned NULL or empty. Returning NA values.")
    prediction <- rep(NA, nrow(x))
  }
  
  prediction <- skcla_as_probability(prediction, object$slevels, object$model$classes_)
  
  return(prediction)
}
