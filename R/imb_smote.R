#'@title Imbalanced Data Handling with SMOTE and RandomForest
#'@description This module applies SMOTE oversampling.
#'@importFrom reticulate source_python r_to_py py_to_r
#'@param random_state Seed for the SMOTE RNG
#'@return A Python SMOTE object
#'@export
create_smote_model <- function(random_state=42) {
  python_path <- system.file("python/imbalanced/smote.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  smote <- inbalanced_create_model(random_state=random_state)
  return(smote)
}

#'@param select_method A SMOTE model (Python object)
#'@param df_train Data frame to resample
#'@param target_column The target column name
#'@return A list (X_resampled, y_resampled)
#'@export
fit_resample_smote <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  df_train_py <- reticulate::r_to_py(df_train)
  list_resample <- select_method$fit_resample(df_train_py, target_column)
  X_train_smote <- reticulate::py_to_r(list_resample[[1]])
  y_train_smote <- reticulate::py_to_r(list_resample[[2]])
  return(list(X_train_smote, y_train_smote))
}