#' @title Feature Selection Using VarianceThreshold
#' @description Applies feature selection based on variance using scikit-learn's VarianceThreshold.
#' @param threshold The variance threshold below which features will be removed
#' @return A Python VarianceThreshold object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @export
create_fe_variance_threshold_model <- function(threshold=0.2) {
  python_path <- system.file(
    "python/feature_select/variance_threshold.py",
    package = "daltoolboxdp"
  )
  reticulate::source_python(python_path)
  vt_model <- fs_create(threshold = threshold)
  return(vt_model)
}

#' @title Fit and Transform with Variance Threshold
#' @description Apply a VarianceThreshold feature selector on a data frame to remove low-variance features.
#' @param select_method A VarianceThreshold model created by `create_fe_variance_threshold_model()`.
#' @param df_train An R data frame of training data.
#' @param target_column Character name of the target column in `df_train`.
#' @return A data frame containing only the features with variance above the specified threshold.
#' @export
fit_transform_fe_variance_threshold <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  python_path <- system.file(
    "python/feature_select/variance_threshold.py",
    package = "daltoolboxdp"
  )
  reticulate::source_python(python_path)

  df_py <- reticulate::r_to_py(df_train)
  X_py <- fit_transform(select_method, df_py, target_column)     # VarianceThreshold ignores target_column
  X_sel <- reticulate::py_to_r(X_py)
  return(X_sel)
}
