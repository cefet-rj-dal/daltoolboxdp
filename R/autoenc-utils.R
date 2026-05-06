# Resolve the canonical epochs argument while preserving temporary compatibility
# with the previous num_epochs name used by autoencoder wrappers.
resolve_autoenc_epochs <- function(epochs, num_epochs = NULL) {
  if (!is.null(num_epochs)) {
    return(as.integer(num_epochs))
  }

  as.integer(epochs)
}
