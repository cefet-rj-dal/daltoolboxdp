#' @title Imbalanced Data Handling with Tomek Links
#' @description This module applies Tomek Links under-sampling followed by RandomForest.
#' @param random_state Seed for the undersampler (if supported)
#' @return A Python TomekLinks object
#' @importFrom reticulate source_python
#' @export
create_imb_tomek_model <- function(random_state=NULL) {
  python_path <- system.file("python/imbalanced/tomek_links.py", package="daltoolboxdp")
  reticulate::source_python(python_path)
  tomek <- inbalanced_create_model(random_state=random_state)
  return(tomek)
}

#' @rdname create_imb_tomek_model
#' @param select_method A TomekLinks model (Python object)
#' @param df_train Data frame to resample
#' @param target_column The target column name as string
#' @return A list (X_resampled, y_resampled)
#' @export
fit_imb_resample_tomek <- function(select_method, df_train, target_column) {
  cat("Column types:", sapply(df_train, class), "\n")
  X_train <- df_train[, !(names(df_train) %in% target_column), drop = FALSE]
  y_train <- df_train[[target_column]]
  list_resample <- select_method$fit_resample(X_train, y_train)
  X_train_tomek <- list_resample[[1]]
  y_train_tomek <- list_resample[[2]]
  return(list(X_train_tomek, y_train_tomek))
}
