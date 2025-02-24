#'@title Feature Selection Using RFE and Logistic Regression
#'@description This module applies Recursive Feature Elimination (RFE) with a LogisticRegression estimator.
#'@import reticulate

#' Create a feature selection model
#'@return A Python RFE object
#'@export
create_rfe_model <- function(n_features_to_select=0.5, lg_max_iter=1000) {
  if (!exists("fs_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/rfe.py")
  }
  rfe_model <- fs_create(n_features_to_select=n_features_to_select, lg_max_iter=lg_max_iter)
  return(rfe_model)
}

#' Fit and transform the dataset using RFE
#'@param select_method The RFE model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string
#'@return Transformed X_train with selected features
#'@export
fit_transform_rfe <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/rfe.py")
  }
  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  # Calling Python function for fitting and transforming
  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  # Convert the result back to an R matrix
 # X_train_selected_r <- reticulate::py_to_r(X_train_selected)

  return(X_train_selected)
}
