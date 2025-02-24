#'@title Feature Selection Using SelectFromModel and RandomForest
#'@description This module applies feature selection using SelectFromModel with a RandomForestClassifier estimator.
#'@import reticulate
#'@return A fitted Python RandomForest model
#'@export
create_fit_rf_model <- function(df_train, target_column, n_estimators=100, random_state=0) {
  reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_rf.py")
  df_train_py <- reticulate::r_to_py(df_train)
  X_train <- df_train_py$drop(target_column, axis=1)$values
  y_train <- df_train_py[[target_column]]$values
  rf_model <- create_rf_model(n_estimators=n_estimators, random_state=random_state, X=X_train, y=y_train)
  return(rf_model)
}

#' Create a feature selection model using SelectFromModel
#'@param model A Python Logistic Regression model
#'@param threshold The threshold for feature selection
#'@param prefit Boolean indicating if the model should be considered prefit
#'@return A Python SelectFromModel object
#'@export
create_fs_model <- function(model, threshold="mean", prefit=TRUE) {
  if (!exists("fs_create")) {
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_rf.py")
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
    reticulate::source_python("daltoolbox/inst/python/sklearn/feature_select/selectf_model_rf.py")
  }

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  X_train_selected <- fit_transform(select_method, df_train_py, target_column)

  return(X_train_selected)
}
