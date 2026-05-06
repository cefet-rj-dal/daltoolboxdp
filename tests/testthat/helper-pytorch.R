library(daltoolboxdp)

.initial_temp_entries <- tryCatch(
  list.files(tempdir(), all.files = TRUE, no.. = TRUE, full.names = TRUE),
  error = function(...) character()
)

.temp_cleanup_env <- new.env(parent = emptyenv())
reg.finalizer(.temp_cleanup_env, function(e) {
  current_entries <- tryCatch(
    list.files(tempdir(), all.files = TRUE, no.. = TRUE, full.names = TRUE),
    error = function(...) character()
  )

  extra_entries <- setdiff(current_entries, .initial_temp_entries)
  extra_entries <- extra_entries[file.exists(extra_entries)]

  if (length(extra_entries) > 0) {
    extra_entries <- extra_entries[order(nchar(extra_entries), decreasing = TRUE)]
    invisible(lapply(extra_entries, function(path) {
      try(unlink(path, recursive = TRUE, force = TRUE), silent = TRUE)
    }))
  }
}, onexit = TRUE)

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
