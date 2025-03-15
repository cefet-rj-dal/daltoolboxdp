#' Combined Sampling
#' 
#' Generate and clean synthetic samples
#' @import reticulate

#' @param sampling_strategy The sampling strategy to use. Default is 'auto'
#' @param random_state Random state for reproducibility
#' @return A Python SMOTETomek object
#' @export
create_smotetomek_model <- function(sampling_strategy='auto', random_state=42) {
  reticulate::source_python("daltoolboxdp/inst/python/sklearn/imbalanced/smote_tomek.py")
  smotetomek <- create_smotetomek_model(sampling_strategy=sampling_strategy, random_state=random_state)
  return(smotetomek)
}

#' Fit and resample dataset using SMOTETomek
#' @param model The SMOTETomek model
#' @param df_train Data frame to resample
#' @param target_column Target column name
#' @return List containing resampled features and target
#' @export
fit_resample <- function(model, df_train, target_column) {
  # Convert df_train to pandas DataFrame
  df_train_py <- reticulate::r_to_py(df_train)
  
  # Separate features and target
  X <- df_train_py$drop(target_column, axis=1)
  y <- df_train_py[[target_column]]
  
  # Perform resampling
  result <- model$fit_resample(X, y)
  
  # Convert back to R objects
  X_resampled <- reticulate::py_to_r(result[[1]])
  y_resampled <- reticulate::py_to_r(result[[2]])
  
  return(list(X_resampled, y_resampled))
}
