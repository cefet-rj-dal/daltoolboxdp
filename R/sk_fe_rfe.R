#' Recursive Feature Elimination
#' 
#' Feature selection using recursive elimination with logistic regression
#' @import reticulate

#' Create RFE model
#' @param n_features_to_select Number/proportion of features to select
#' @param max_iter Maximum iterations for logistic regression
#' @return A Python RFE object
#' @export
create_rfe_model <- function(n_features_to_select=0.5, max_iter=1000) {
  reticulate::source_python("inst/python/sklearn/feature_select/rfe.py")
  model <- fs_create(n_features_to_select=n_features_to_select, 
                    lg_max_iter=max_iter)
  return(model)
}

#'@param model The RFE model object
#'@param df_train Input data frame
#'@param target_column Name of target column
#'@return Transformed feature matrix
#'@export
fit_transform <- function(model, df_train, target_column) {
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  result <- model$fit_transform(X, y)
  return(reticulate::py_to_r(result))
}
