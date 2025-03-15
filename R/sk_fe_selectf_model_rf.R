#' Random Forest Feature Selection
#' 
#' Feature selection using random forest classifier as estimator
#' @import reticulate

#' Create random forest feature selector
#' @param n_estimators Number of trees (default: 100)
#' @param threshold Threshold for feature selection
#' @return A Python SelectFromModel object
#' @export
create_rf_select_model <- function(n_estimators=100, threshold="mean") {
  reticulate::source_python("inst/python/sklearn/feature_select/selectf_model_rf.py")
  model <- create_rf_select_model(n_estimators=n_estimators, threshold=threshold)
  return(model)
}

#' Fit and transform the dataset
#' @param model The RandomForest selection model
#' @param df_train Data frame to transform
#' @param target_column Target column name
#' @return Transformed features
#' @export
fit_transform <- function(model, df_train, target_column) {
  df_train_py <- reticulate::r_to_py(df_train)
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  result <- model$fit_transform(X, y)
  return(reticulate::py_to_r(result))
}
