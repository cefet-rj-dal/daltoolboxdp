#'@title Feature Selection Using SelectFromModel and Logistic Regression
#'@description This module applies feature selection using SelectFromModel with a LogisticRegression estimator.
#'@import reticulate
#'@importFrom reticulate source_python r_to_py py_to_r

#' Create and fit a logistic regression model
#'@return A fitted Python Logistic Regression model
#'@export
create_fit_lg_model <- function(df_train, target_column, C=0.1, penalty='l1', solver='liblinear') {
  python_path <- system.file("python/sklearn/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X <- df_py$drop(target_column, axis=1)$values
  y <- df_py[[target_column]]$values
  lg_model <- create_lg_model(C=C, penalty=penalty, solver=solver, X=X, y=y)
  return(lg_model)
}

#' Create a feature selection model using SelectFromModel
#'@param model A Python Logistic Regression model
#'@param threshold The threshold for feature selection
#'@param prefit Boolean indicating if the model should be considered prefit
#'@return A Python SelectFromModel object
#'@export
create_fs_model <- function(model, threshold="mean", prefit=TRUE) {
  python_path <- system.file("python/sklearn/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  sf_method <- fs_create(model, threshold=threshold, prefit=prefit)
  return(sf_method)
}

#' Fit and transform the dataset using the feature selection model
#'@param select_method The feature selection model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string
#'@return Transformed X_train with selected features
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
