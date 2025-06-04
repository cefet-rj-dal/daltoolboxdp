#' @title Imbalanced Data Handling with SMOTE and RandomForest
#' @description This module applies SMOTE oversampling.
#' @param random_state Seed for the SMOTE RNG
#' @return A Python SMOTE object
#' @importFrom reticulate source_python r_to_py py_to_r
#' @rdname create_imb_smote_model
#' @export
create_imb_smote_model <- function(random_state=42) {
  python_path <- system.file("python/imbalanced/smote.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  smote <- inbalanced_create_model(random_state=random_state)
  return(smote)
}

#' @rdname create_imb_smote_model
#' @param select_method A SMOTE model (Python object)
#' @param df_train Data frame to resample
#' @param target_column The target column name as string
#' @return A list (X_resampled, y_resampled)
#' @export
fit_imb_resample_smote <- function(select_method, df_train, target_column) {
  python_path <- system.file("python/imbalanced/smote.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  cat("Column types:", sapply(df_train, class), "\n")
  df_train_py <- reticulate::r_to_py(df_train)
  list_resample <- fit_resample(select_method, df_train_py, target_column)
  X_train_smote <- reticulate::py_to_r(list_resample[[1]])
  y_train_smote <- reticulate::py_to_r(list_resample[[2]])
  return(list(X_train_smote, y_train_smote))
}