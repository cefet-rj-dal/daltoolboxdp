#' @title Imbalanced Data Handling with SMOTETomek
#' @description This module applies SMOTETomek for handling imbalanced data.
#' @param random_state Seed for the random number generator
#' @return A Python SMOTETomek object
#' @importFrom reticulate source_python r_to_py
#' @export
create_skimb_smote_tomek_model <- function(random_state=42) {
  # source the python implementation from the installed package
  python_path <- system.file("python/imbalanced/smote_tomek_links.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  stomek <- inbalanced_create_model(random_state=random_state)
  return(stomek)
}

#' @rdname create_skimb_smote_tomek_model
#' @param select_method A SMOTETomek model (Python object)
#' @param df_train Data frame to resample
#' @param target_column The target column name as string
#' @return List containing resampled X_train and y_train
#' @importFrom reticulate py_to_r
#' @export
fit_imb_resample_smotetomek <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  df_train_py <- reticulate::r_to_py(df_train)
  X_train <- df_train_py$drop(target_column, axis=1)$values
  y_train <- df_train_py[[target_column]]$values
  list_resample <- select_method$fit_resample(X_train, y_train)
  X_train_smotetomek <- list_resample[[1]]
  y_train_smotetomek <- list_resample[[2]]
  return(list(X_train_smotetomek, y_train_smotetomek))
}
