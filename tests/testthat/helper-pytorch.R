library(daltoolboxdp)

skip_if_no_python_module <- function(module_name) {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("reticulate")

  py_ready <- tryCatch(
    reticulate::py_available(initialize = TRUE),
    error = function(...) FALSE
  )
  if (!py_ready) {
    testthat::skip("Python is not available for reticulate.")
  }

  module_ready <- tryCatch(
    reticulate::py_module_available(module_name),
    error = function(...) FALSE
  )
  if (!module_ready) {
    testthat::skip(paste0("Python module '", module_name, "' is not available in the active Python environment."))
  }
}

skip_if_no_pytorch <- function() {
  skip_if_no_python_module("torch")
}

skip_if_no_sklearn <- function() {
  skip_if_no_python_module("sklearn")
}
