prepare_skcla_fit <- function(obj, data) {
  data <- adjust_data.frame(data)
  data[, obj$attribute] <- adjust_factor(data[, obj$attribute], obj$ilevels, obj$slevels)
  obj$x <- setdiff(colnames(data), obj$attribute)
  list(obj = obj, data = data)
}

prepare_skcla_predict_data <- function(obj, x) {
  x <- adjust_data.frame(x)
  if (!is.null(obj$x)) {
    return(x[, obj$x, drop = FALSE])
  }
  x[, !names(x) %in% obj$attribute, drop = FALSE]
}

skcla_as_probability <- function(prediction, slevels, classes = NULL) {
  if (is.null(prediction) || length(prediction) == 0) {
    empty <- matrix(numeric(0), nrow = 0, ncol = length(slevels))
    colnames(empty) <- slevels
    return(as.data.frame(empty))
  }

  if (is.data.frame(prediction) || is.matrix(prediction)) {
    probs <- as.matrix(prediction)
  } else if (is.atomic(prediction) && !is.list(prediction)) {
    return(adjust_class_label(factor(prediction, levels = slevels)) |> as.data.frame())
  } else {
    probs <- tryCatch(
      {
        if (is.list(prediction)) {
          do.call(rbind, prediction)
        } else {
          matrix(prediction, ncol = length(slevels), byrow = TRUE)
        }
      },
      error = function(cond) {
        NULL
      }
    )
  }

  if (is.null(probs)) {
    return(adjust_class_label(factor(prediction, levels = slevels)) |> as.data.frame())
  }

  probs <- as.matrix(probs)
  if (!is.null(classes) && length(classes) == ncol(probs)) {
    colnames(probs) <- as.character(classes)
  }
  if (is.null(colnames(probs)) && ncol(probs) == length(slevels)) {
    colnames(probs) <- slevels
  }
  if (!is.null(colnames(probs))) {
    probs <- probs[, slevels, drop = FALSE]
  }
  as.data.frame(probs)
}
