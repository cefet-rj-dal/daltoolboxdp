library(testthat)
library(daltoolboxdp)

.temp_root <- normalizePath(dirname(tempdir()), winslash = "/", mustWork = FALSE)
.initial_temp_root_entries <- tryCatch(
  list.files(.temp_root, all.files = TRUE, no.. = TRUE, full.names = TRUE),
  error = function(...) character()
)

cleanup_check_tempdir <- function() {
  current_entries <- tryCatch(
    list.files(.temp_root, all.files = TRUE, no.. = TRUE, full.names = TRUE),
    error = function(...) character()
  )

  extra_entries <- setdiff(current_entries, .initial_temp_root_entries)
  extra_entries <- extra_entries[file.exists(extra_entries)]

  if (length(extra_entries) == 0) {
    return(invisible(NULL))
  }

  base_names <- basename(extra_entries)
  extra_entries <- extra_entries[grepl("^tmp", base_names)]

  if (length(extra_entries) == 0) {
    return(invisible(NULL))
  }

  extra_entries <- extra_entries[order(nchar(extra_entries), decreasing = TRUE)]
  invisible(lapply(extra_entries, function(path) {
    try(unlink(path, recursive = TRUE, force = TRUE), silent = TRUE)
  }))
}

on.exit(cleanup_check_tempdir(), add = TRUE)

test_check("daltoolboxdp")
