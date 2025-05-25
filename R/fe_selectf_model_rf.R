#'@title Feature Selection Using SelectFromModel and RandomForest
#'@description This module applies feature selection using SelectFromModel with a RandomForestClassifier estimator.
#'@importFrom reticulate source_python r_to_py py_to_r
#'@return A fitted Python RandomForest model
#'@export
create_fit_rf_model <- function(df_train, target_column, n_estimators=100, random_state=0) {
  python_path <- system.file("python/sklearn/feature_select/selectf_model_rf.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X <- df_py$drop(target_column, axis=1)$values
  y <- df_py[[target_column]]$values
  rf_model <- create_rf_model(n_estimators=n_estimators, random_state=random_state, X=X, y=y)
  return(rf_model)
}

#' @describeIn create_fit_rf_model Create a feature selection model using SelectFromModel
#'@param model A Python Logistic Regression model
#'@param threshold The threshold for feature selection
#'@param prefit Boolean indicating if the model should be considered prefit
#'@return A Python SelectFromModel object
#'@export
create_fs_model <- function(model, threshold="mean", prefit=TRUE) {
  python_path <- system.file("python/sklearn/feature_select/selectf_model_rf.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  sf_method <- fs_create(model, threshold=threshold, prefit=prefit)
  return(sf_method)
}

#' @describeIn create_fit_rf_model Fit and transform the dataset using the feature selection model
#'@param select_method The feature selection model (Python object)
#'@param df_train Data frame to transform
#'@param target_column The target column name as string
#'@return Transformed X_train with selected features
#'@export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/selectf_model_rf.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
