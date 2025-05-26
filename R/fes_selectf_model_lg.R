#' @title SelectFromModel + Logistic Regression
#' @description Apply SelectFromModel using a LogisticRegression estimator.
#' @param df_train Data frame to fit the LogisticRegression model
#' @param target_column The target column name as string
#' @param C Inverse of regularization strength
#' @param penalty Norm used in the penalization ("l1", "l2", etc.)
#' @param solver Algorithm to use in optimization ("liblinear", …)
#' @return A fitted Python Logistic Regression model
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_fe_lg_model <- function(df_train, target_column, C=0.1, penalty='l1', solver='liblinear') {
  python_path <- system.file("python/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X <- df_py$drop(target_column, axis=1)$values
  y <- df_py[[target_column]]$values
  lg_model <- create_lg_model(C=C, penalty=penalty, solver=solver, X=X, y=y)
  return(lg_model)
}

#' @describeIn create_fe_lg_model Create a feature selector using SelectFromModel
#' @param model A prefit Python model
#' @param threshold Threshold for feature selection (e.g. "mean")
#' @param prefit Logical; whether the model is already fitted
#' @return A Python SelectFromModel object
#' @export
create_fe_selectfrommodel_lg <- function(model, threshold="mean", prefit=TRUE) {
  python_path <- system.file("python/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  sf_method <- fs_create(model, threshold=threshold, prefit=prefit)
  return(sf_method)
}

#' @describeIn create_fe_lg_model Fit and transform using SelectFromModel LG
#' @param select_method The feature selection model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string
#' @return Transformed X_train with selected features
#' @export
fit_transform_fe_selectfrommodel_lg <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/selectf_model_lg.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  df_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_py, target_column)
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
