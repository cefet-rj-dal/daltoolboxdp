#'@title Feature Selection Using SelectFromModel and Logistic Regression
#'@description This module applies feature selection using SelectFromModel with a LogisticRegression estimator.
#'@import reticulate

#' Create and fit a logistic regression model
#'@return A fitted Python Logistic Regression model
#'@export
create_fit_lg_model <- function(df_train, target_column, C=0.1, penalty='l1', solver='liblinear') {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_lg.py")
  df_train_py <- reticulate::r_to_py(df_train)
  X_train <- df_train_py$drop(target_column, axis=1)$values
  y_train <- df_train_py[[target_column]]$values
  lg_model <- create_lg_model(C=C, penalty=penalty, solver=solver, X=X_train, y=y_train)
  return(lg_model)
}

#' Create a feature selection model using SelectFromModel
#'@param model A Python Logistic Regression model
#'@param threshold The threshold for feature selection
#'@param prefit Boolean indicating if the model should be considered prefit
#'@return A Python SelectFromModel object
#'@export
create_fs_model <- function(model, threshold="mean", prefit=TRUE) {
  if (!exists("fs_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_lg.py")
  }
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
  if (!exists("fit_transform")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_lg.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  return(X_train_selected)
}
