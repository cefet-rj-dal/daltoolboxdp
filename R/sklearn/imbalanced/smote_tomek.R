#'@title Imbalanced Data Handling with SMOTETomek
#'@description This module applies SMOTETomek for handling imbalanced data.
#'@import reticulate

#' Create a SMOTETomek model
#'@param random_state The seed used by the random number generator
#'@return A Python SMOTETomek object
#'@export
create_smotetomek_model <- function(random_state=42) {
  reticulate::source_python("path_to_your_python_script.py")
  stomek <- inbalanced_create_model(random_state=random_state)
  return(stomek)
}

#' Fit and resample the dataset using SMOTETomek
#'@param select_method The SMOTETomek model (Python object)
#'@param df_train Data frame to resample
#'@param target_column The target column name as string
#'@return List containing resampled X_train and y_train
#'@export
fit_resample_smotetomek <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")

  # Convert df_train to a pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)

  # Extract X and y
  X_train <- df_train_py$drop(target_column, axis=1)$values
  y_train <- df_train_py[[target_column]]$values

  # Calling Python function for resampling
  list_resample <- select_method$fit_resample(X_train, y_train)

  X_train_smotetomek<- list_resample[[1]]
  y_train_smotetomek <- list_resample[[2]]

  return(list(X_train_smotetomek, y_train_smotetomek))
}
