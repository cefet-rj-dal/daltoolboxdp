#'@title Feature Selection Using Logistic Regression
#'@description Feature selection using LogisticRegression as estimator.
#'@import reticulate

#' Create LogisticRegression feature selector
#'@param C Inverse regularization strength
#'@param threshold Feature selection threshold
#'@return A Python SelectFromModel object
#'@export
create_lg_select_model <- function(C=0.1, threshold="mean") {
  reticulate::source_python("inst/python/sklearn/feature_select/selectf_model_lg.py")
  model <- create_lg_select_model(C=C, threshold=threshold)
  return(model)
}

#' Fit and transform the dataset
#'@param model The LogisticRegression selection model
#'@param df_train Data frame to transform
#'@param target_column Target column name
#'@return Transformed features
#'@export
fit_transform <- function(model, df_train, target_column) {
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  result <- model$fit_transform(X, y)
  return(reticulate::py_to_r(result))
}
