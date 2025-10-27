#'@title LASSO Feature Selection
#'@description Performs feature selection using L1-regularized regression (LASSO),
#' implemented with `glmnet`.
#'
#'@param attribute Character. Name of the target variable.
#'@return A `fs_lasso` object.
#'
#'@references
#' Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
#'
#'@examples
#'\dontrun{
#'data(iris)
#'
#'# 1) LASSO requires a numeric response
#'fs <- daltoolbox::fit(fs_lasso("Sepal.Length"), iris)
#'fs$features                 # selected predictors with non-zero coefficients
#'
#'# 2) Subset data to selected features + target
#'data_lasso <- daltoolbox::transform(fs, iris)
#'head(data_lasso)
#'}
#'@importFrom daltoolbox dal_transform
#'@importFrom daltoolbox fit
#'@importFrom daltoolbox transform
#'@export
fs_lasso <- function(attribute) {
  obj <- fs(attribute)
  class(obj) <- append("fs_lasso", class(obj))
  return(obj)
}


#'@importFrom daltoolbox fit
#'@importFrom glmnet cv.glmnet
#'@importFrom glmnet glmnet
#'@export
fit.fs_lasso <- function(obj, data, ...) {
  data = data.frame(data)
  if (!is.numeric(data[,obj$attribute]))
    data[,obj$attribute] =  as.numeric(data[,obj$attribute])

  nums = unlist(lapply(data, is.numeric))
  data = data[ , nums]

  predictors_name  = setdiff(colnames(data), obj$attribute)
  predictors = as.matrix(data[,predictors_name])
  predictand = data[,obj$attribute]
  grid = 10^seq(10, -2, length = 100)
  cv.out = glmnet::cv.glmnet(predictors, predictand, alpha = 1)
  bestlam = cv.out$lambda.min
  out = glmnet::glmnet(predictors, predictand, alpha = 1, lambda = grid)
  lasso.coef = predict(out,type = "coefficients", s = bestlam)
  l = lasso.coef[(lasso.coef[,1]) != 0,0]
  vec = rownames(l)[-1]

  obj$features <- vec

  return(obj)
}

