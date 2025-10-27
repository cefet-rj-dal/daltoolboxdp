#' Neural Network Classifier
#'@title Multi-layer Perceptron Classifier
#'@description Implements classification using a multi-layer perceptron (MLP).
#' Wraps scikit-learn's `MLPClassifier` through `reticulate`.
#'@param attribute Target attribute name for model building
#'@param slevels List of possible values for classification target
#'@param hidden_layer_sizes Number of neurons in each hidden layer
#'@param activation Activation function for hidden layer ('identity', 'logistic', 'tanh', 'relu')
#'@param solver The solver for weight optimization ('lbfgs', 'sgd', 'adam')
#'@param alpha L2 penalty (regularization term) parameter
#'@param batch_size Size of minibatches for stochastic optimizers
#'@param learning_rate Learning rate schedule for weight updates
#'@param learning_rate_init Initial learning rate used
#'@param power_t Exponent for inverse scaling learning rate
#'@param max_iter Maximum number of iterations
#'@param shuffle Whether to shuffle samples in each iteration
#'@param random_state Seed for random number generation
#'@param tol Tolerance for optimization
#'@param verbose Whether to print progress messages to stdout
#'@param warm_start Whether to reuse previous solution
#'@param momentum Momentum for gradient descent update
#'@param nesterovs_momentum Whether to use Nesterov's momentum
#'@param early_stopping Whether to use early stopping
#'@param validation_fraction Proportion of training data for validation
#'@param beta_1 Exponential decay rate for estimates of first moment vector
#'@param beta_2 Exponential decay rate for estimates of second moment vector
#'@param epsilon Value for numerical stability in adam
#'@param n_iter_no_change Maximum number of epochs to not meet tol improvement
#'@param max_fun Maximum number of loss function calls
#'@return A `skcla_mlp` classifier object.
#'
#'@references
#' Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
#'@examples
#'\dontrun{
#'data(iris)
#'clf <- skcla_mlp(attribute = 'Species', slevels = levels(iris$Species), hidden_layer_sizes = c(32, 16))
#'clf <- daltoolbox::fit(clf, iris)
#'pred <- predict(clf, iris)
#'table(pred, iris$Species)
#'}
#'
#'# More examples:
#'# https://github.com/cefet-rj-dal/daltoolboxdp/blob/main/examples/skcla_mlp.md
#'@import daltoolbox
#'@export
skcla_mlp <- function(attribute, slevels,
                    hidden_layer_sizes = c(100),
                    activation = 'relu',
                    solver = 'adam',
                    alpha = 0.0001,
                    batch_size = 'auto',
                    learning_rate = 'constant',
                    learning_rate_init = 0.001,
                    power_t = 0.5,
                    max_iter = 200,
                    shuffle = TRUE,
                    random_state = NULL,
                    tol = 1e-4,
                    verbose = FALSE,
                    warm_start = FALSE,
                    momentum = 0.9,
                    nesterovs_momentum = TRUE,
                    early_stopping = FALSE,
                    validation_fraction = 0.1,
                    beta_1 = 0.9,
                    beta_2 = 0.999,
                    epsilon = 1e-8,
                    n_iter_no_change = 10,
                    max_fun = 15000) {
  obj <- classification(attribute, slevels)
  cobj <- class(obj)
  objex <- list(
    hidden_layer_sizes = as.integer(hidden_layer_sizes),
    activation = activation,
    solver = solver,
    alpha = as.numeric(alpha),
    batch_size = batch_size,
    learning_rate = learning_rate,
    learning_rate_init = as.numeric(learning_rate_init),
    power_t = as.numeric(power_t),
    max_iter = as.integer(max_iter),
    shuffle = shuffle,
    random_state = if(!is.null(random_state)) as.integer(random_state) else NULL,
    tol = as.numeric(tol),
    verbose = verbose,
    warm_start = warm_start,
    momentum = as.numeric(momentum),
    nesterovs_momentum = nesterovs_momentum,
    early_stopping = early_stopping,
    validation_fraction = as.numeric(validation_fraction),
    beta_1 = as.numeric(beta_1),
    beta_2 = as.numeric(beta_2),
    epsilon = as.numeric(epsilon),
    n_iter_no_change = as.integer(n_iter_no_change),
    max_fun = as.integer(max_fun)
  )
  
  obj <- c(obj, objex)
  class(obj) <- c("skcla_mlp", cobj)
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@exportS3Method fit skcla_mlp
fit.skcla_mlp <- function(obj, data, ...) {
  python_path <- system.file("python/skcla_mlp.py", package = "daltoolboxdp")
  if (!file.exists(python_path)) {
    stop("Python source file not found. Please check package installation.")
  }
  reticulate::source_python(python_path)
  
  if (is.null(obj$model)) {
    obj$model <- skcla_mlp_create(
      hidden_layer_sizes = obj$hidden_layer_sizes,
      activation = obj$activation,
      solver = obj$solver,
      alpha = obj$alpha,
      batch_size = obj$batch_size,
      learning_rate = obj$learning_rate,
      learning_rate_init = obj$learning_rate_init,
      power_t = obj$power_t,
      max_iter = obj$max_iter,
      shuffle = obj$shuffle,
      random_state = obj$random_state,
      tol = obj$tol,
      verbose = obj$verbose,
      warm_start = obj$warm_start,
      momentum = obj$momentum,
      nesterovs_momentum = obj$nesterovs_momentum,
      early_stopping = obj$early_stopping,
      validation_fraction = obj$validation_fraction,
      beta_1 = obj$beta_1,
      beta_2 = obj$beta_2,
      epsilon = obj$epsilon,
      n_iter_no_change = obj$n_iter_no_change,
      max_fun = obj$max_fun
    )
  }
  
  data <- adjust_data.frame(data)
  obj$model <- skcla_mlp_fit(obj$model, data, obj$attribute)
  
  return(obj)
}

#'@import daltoolbox
#'@import reticulate
#'@export
predict.skcla_mlp <- function(object, x, ...) {
  if (!exists("skcla_mlp_predict")) {
    python_path <- system.file("python/skcla_mlp.py", package = "daltoolboxdp")
    if (!file.exists(python_path)) {
      stop("Python source file not found. Please check package installation.")
    }
    reticulate::source_python(python_path)
  }
  
  # Prepare features for prediction
  x <- adjust_data.frame(x)
  x <- x[, !names(x) %in% object$attribute]
  
  prediction <- skcla_mlp_predict(object$model, x)
  prediction <- adjust_class_label(prediction)
  
  return(prediction)
}
