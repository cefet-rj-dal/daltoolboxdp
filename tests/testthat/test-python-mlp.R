make_reg_data <- function(n = 20L) {
  data.frame(
    x1 = seq_len(n),
    x2 = seq_len(n) / 2,
    target = seq_len(n) * 0.1
  )
}

make_cla_data <- function() {
  data.frame(
    x1 = c(0, 0, 1, 1, 0, 1, 0, 1),
    x2 = c(0, 1, 0, 1, 0, 1, 1, 0),
    class = factor(c("a", "a", "b", "b", "a", "b", "a", "b"))
  )
}

test_that("torch_reg_mlp constructor exposes unified defaults", {
  model <- torch_reg_mlp(attribute = "target", input_size = 2L, hidden_sizes = c(4L, 2L))

  expect_s3_class(model, "torch_reg_mlp")
  expect_identical(model$epochs, 100L)
  expect_identical(model$validation_strategy, "static")
  expect_identical(model$stopping_rule, "none")
})

test_that("torch_reg_mlp fits and predicts through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_reg_data()
  model <- torch_reg_mlp(
    attribute = "target",
    input_size = 2L,
    hidden_sizes = c(4L, 2L),
    epochs = 2L
  )
  fitted <- fit.torch_reg_mlp(model, df)
  pred <- predict.torch_reg_mlp(fitted, df[, c("x1", "x2")])

  expect_length(pred, nrow(df))
  expect_true(length(fitted$train_loss_hist) >= 1)
  expect_true(!is.null(fitted$model))
})

test_that("torch_cla_mlp constructor exposes unified defaults", {
  model <- torch_cla_mlp(
    attribute = "class",
    slevels = c("a", "b"),
    input_size = 2L,
    hidden_sizes = c(4L)
  )

  expect_s3_class(model, "torch_cla_mlp")
  expect_identical(model$epochs, 100L)
  expect_identical(model$validation_strategy, "static")
  expect_identical(model$stopping_rule, "none")
})

test_that("torch_cla_mlp fits and predicts through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_cla_data()
  model <- torch_cla_mlp(
    attribute = "class",
    slevels = c("a", "b"),
    input_size = 2L,
    hidden_sizes = c(4L),
    epochs = 2L,
    validation_strategy = "dynamic",
    stopping_rule = "patience"
  )
  fitted <- fit.torch_cla_mlp(model, df)
  pred <- predict.torch_cla_mlp(fitted, df[, c("x1", "x2")])
  probs <- predict_proba.torch_cla_mlp(fitted, df[, c("x1", "x2")])

  expect_equal(nrow(pred), nrow(df))
  expect_true(ncol(pred) >= 1L)
  expect_length(probs, nrow(df))
  expect_true(all(vapply(probs, length, integer(1)) == 2L))
  expect_true(length(fitted$train_loss_hist) >= 1)
  expect_true(!is.null(fitted$model))
})
