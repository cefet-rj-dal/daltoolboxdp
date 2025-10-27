#'@title Forward Stepwise Selection
#'@description Greedy feature selection that iteratively adds the feature which most improves the
#' model according to an adjustment metric (e.g., adjusted R^2). Wraps `leaps::regsubsets`.
#'
#'@param attribute Character. Name of the target variable.
#'@return A `fs_fss` object.
#'
#'@references
#' Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
#'
#'@examples
#'\dontrun{
#'data(iris)
#'
#'# 1) Forward stepwise for numeric response (adjusted R^2 criterion)
#'fs <- daltoolbox::fit(fs_fss("Sepal.Length"), iris)
#'fs$features
#'
#'# 2) Subset to selected features + target
#'data_fss <- daltoolbox::transform(fs, iris)
#'head(data_fss)
#'}
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
fs_fss <- function(attribute) {
  obj <- fs(attribute)
  class(obj) <- append("fs_fss", class(obj))
  return(obj)
}

#'@importFrom daltoolbox fit
#'@importFrom stats coef
#'@export
fit.fs_fss <- function(obj, data, ...) {
  data = data.frame(data)
  if (!is.numeric(data[, obj$attribute]))
    data[, obj$attribute] = as.numeric(data[, obj$attribute])

  nums = unlist(lapply(data, is.numeric))
  data = data[, nums]

  predictors_name = setdiff(colnames(data), obj$attribute)
  predictors = as.matrix(data[, predictors_name])
  predictand = data[, obj$attribute]

  # Run forward stepwise selection
  regfit.fwd = leaps::regsubsets(predictors, predictand, nvmax = ncol(data) - 1, method = "forward")
  reg.summaryfwd = summary(regfit.fwd)
  b1 = which.max(reg.summaryfwd$adjr2)
  t = stats::coef(regfit.fwd, b1)
  vec = names(t)[-1]

  obj$features <- vec

  return(obj)
}


