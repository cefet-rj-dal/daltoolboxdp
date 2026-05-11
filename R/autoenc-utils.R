# Resolve the canonical epochs argument while preserving temporary compatibility
# with the previous num_epochs name used by autoencoder wrappers.
resolve_autoenc_epochs <- function(epochs, num_epochs = NULL) {
  if (!is.null(num_epochs)) {
    return(as.integer(num_epochs))
  }

  as.integer(epochs)
}

normalize_hidden_sizes <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  as.list(as.integer(x))
}

normalize_stage_hidden_sizes <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  if (is.list(x)) {
    return(lapply(x, normalize_hidden_sizes))
  }

  normalize_hidden_sizes(x)
}

normalize_encoding_sizes <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  as.integer(x)
}
