#' @title Feature Selection Using RFE and Logistic Regression
#' @description This module applies Recursive Feature Elimination (RFE) with a LogisticRegression estimator.
#' @param n_features_to_select Number or fraction of features to select
#' @param lg_max_iter Maximum number of iterations for the LogisticRegression estimator
#' @return A Python RFE object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_rfe_model <- function(n_features_to_select=0.5, lg_max_iter=1000) {
  python_path <- system.file("python/sklearn/feature_select/rfe.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  rfe_model <- fs_create(n_features_to_select=n_features_to_select, lg_max_iter=lg_max_iter)
  return(rfe_model)
}

#' @describeIn create_rfe_model Fit and transform the dataset using RFE
#' @param select_method The RFE model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string
#' @return Transformed X_train with selected features
#' @export
fit_transform_rfe <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/rfe.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_train_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_train_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
