#'@title Naive Bayes Classifier
#'@description Classifies using the Naive Bayes algorithm.
#' It wraps the sklearn library.
#'@param attribute attribute target to model building.
#'@param slevels Possible values for the target classification.
#'@param priors Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
#'@param var_smoothing Portion of the largest variance of all features that is added to variances for stability.
#'@return A nb object.
#'@examples
#'data(iris)
#'slevels <- levels(iris$Species)
#'model <- cla_nb("Species", slevels)
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
cla_nb <- function(attribute, slevels,
                   priors = NULL,
                   var_smoothing = 1e-9) {
  obj <- list(
    attribute = attribute,
    slevels = slevels,
    priors = priors,
    var_smoothing = as.numeric(var_smoothing)
  )

  class(obj) <- c("cla_nb", class(obj))
  return(obj)
}

#'@import reticulate
#'@export
fit.cla_nb <- function(obj, data, ...) {
  # Source the Python file only if the function does not already exist
  if (!exists("nb_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_nb.py")
  }

  # Check if the model is already initialized, otherwise create it
  if (is.null(obj$model)) {
    obj$model <- nb_create(
      obj$priors,
      obj$var_smoothing
    )
  }

  # Adjust the data frame if needed
  data <- adjust_data.frame(data)

  # Fit the model using the Naive Bayes function and the attributes from obj
  obj$model <- nb_fit(obj$model, data, obj$attribute)

  return(obj)
}

#'@import reticulate
#'@export
predict.cla_nb <- function(obj, data, ...) {
  if (!exists("nb_predict"))
    reticulate::source_python("daltoolbox/inst/python/sklearn/cla_nb.py")

  data <- adjust_data.frame(data)
  data <- data[, !names(data) %in% obj$attribute]

  prediction <- nb_predict(obj$model, data)
  prediction <- adjust_class_label(prediction)

  return(prediction)
}
