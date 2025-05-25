#' @title Feature Selection Using Smote
#' @description This module applies feature selection based on variance using Smote.
#' @param random_state Seed for the SMOTE model
#' @return A Python Smote object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_smote_model <- function(random_state=42) {
  python_path <- system.file("python/sklearn/feature_select/smote.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  sf_method <- inbalanced_create_model(random_state=random_state)
  return(sf_method)
}

#' @describeIn create_smote_model Fit and transform the dataset using Smote
#' @param select_method The Smote model (Python object)
#' @param df_train Data frame to transform
#' @param target_column The target column name as string (not used for fitting)
#' @return A list of resampled (X_res, y_res)
#' @export
fit_transform_fs <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file("python/sklearn/feature_select/smote.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  res_py <- fit_resample(select_method, df_train, target_column)
  X_py <- res_py[[1]]; y_py <- res_py[[2]]
  X_res <- reticulate::py_to_r(X_py); y_res <- reticulate::py_to_r(y_py)
  return(list(X_res, y_res))
}
