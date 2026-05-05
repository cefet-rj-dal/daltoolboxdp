make_autoenc_data <- function(n = 16L, p = 5L) {
  as.data.frame(matrix(runif(n * p), nrow = n, ncol = p))
}

expect_autoenc_defaults <- function(model, cls) {
  expect_s3_class(model, cls)
  expect_identical(model$num_epochs, 100L)
  expect_identical(model$validation_strategy, "static")
  expect_identical(model$stopping_rule, "none")
}

test_that("autoencoder constructors expose unified defaults", {
  expect_autoenc_defaults(autoenc_e(5L, 2L), "autoenc_e")
  expect_autoenc_defaults(autoenc_ed(5L, 2L), "autoenc_ed")
  expect_autoenc_defaults(autoenc_conv_e(5L, 2L), "autoenc_conv_e")
  expect_autoenc_defaults(autoenc_conv_ed(5L, 2L), "autoenc_conv_ed")
  expect_autoenc_defaults(autoenc_lstm_e(5L, 2L), "autoenc_lstm_e")
  expect_autoenc_defaults(autoenc_lstm_ed(5L, 2L), "autoenc_lstm_ed")
  expect_autoenc_defaults(autoenc_denoise_e(5L, 2L), "autoenc_denoise_e")
  expect_autoenc_defaults(autoenc_denoise_ed(5L, 2L), "autoenc_denoise_ed")
  expect_autoenc_defaults(autoenc_variational_e(5L, 2L), "autoenc_variational_e")
  expect_autoenc_defaults(autoenc_variational_ed(5L, 2L), "autoenc_variational_ed")
  expect_autoenc_defaults(autoenc_stacked_e(5L, 2L), "autoenc_stacked_e")
  expect_autoenc_defaults(autoenc_stacked_ed(5L, 2L), "autoenc_stacked_ed")
  expect_autoenc_defaults(autoenc_adv_e(5L, 2L), "autoenc_adv_e")
  expect_autoenc_defaults(autoenc_adv_ed(5L, 2L), "autoenc_adv_ed")
})

test_that("dense autoencoders fit and transform through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_autoenc_data()

  enc <- autoenc_e(input_size = 5L, encoding_size = 2L, num_epochs = 2L)
  enc_fit <- fit.autoenc_e(enc, df)
  enc_out <- transform.autoenc_e(enc_fit, df)

  dec <- autoenc_ed(
    input_size = 5L,
    encoding_size = 2L,
    num_epochs = 2L,
    validation_strategy = "dynamic",
    stopping_rule = "patience"
  )
  dec_fit <- fit.autoenc_ed(dec, df)
  dec_out <- transform.autoenc_ed(dec_fit, df)

  expect_equal(dim(enc_out), c(nrow(df), 2L))
  expect_equal(dim(dec_out), c(nrow(df), ncol(df)))
})

test_that("conv and lstm autoencoders fit and transform through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_autoenc_data()

  conv_enc <- autoenc_conv_e(
    input_size = 5L,
    encoding_size = 2L,
    num_epochs = 2L,
    validation_strategy = "dynamic",
    stopping_rule = "ema"
  )
  conv_enc_fit <- fit.autoenc_conv_e(conv_enc, df)
  conv_enc_out <- transform.autoenc_conv_e(conv_enc_fit, df)

  conv_dec <- autoenc_conv_ed(input_size = 5L, encoding_size = 2L, num_epochs = 2L)
  conv_dec_fit <- fit.autoenc_conv_ed(conv_dec, df)
  conv_dec_out <- transform.autoenc_conv_ed(conv_dec_fit, df)

  lstm_enc <- autoenc_lstm_e(input_size = 5L, encoding_size = 2L, num_epochs = 2L)
  lstm_enc_fit <- fit.autoenc_lstm_e(lstm_enc, df)
  lstm_enc_out <- transform.autoenc_lstm_e(lstm_enc_fit, df)

  lstm_dec <- autoenc_lstm_ed(input_size = 5L, encoding_size = 2L, num_epochs = 2L)
  lstm_dec_fit <- fit.autoenc_lstm_ed(lstm_dec, df)
  lstm_dec_out <- transform.autoenc_lstm_ed(lstm_dec_fit, df)

  expect_equal(dim(conv_enc_out), c(nrow(df), 2L))
  expect_equal(dim(conv_dec_out), c(nrow(df), ncol(df)))
  expect_equal(dim(lstm_enc_out), c(nrow(df), 2L))
  expect_equal(dim(lstm_dec_out), c(nrow(df), ncol(df)))
})

test_that("denoise and variational autoencoders fit and transform through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_autoenc_data()

  dns_enc <- autoenc_denoise_e(input_size = 5L, encoding_size = 2L, noise_factor = 0.1, num_epochs = 2L)
  dns_enc_fit <- fit.autoenc_denoise_e(dns_enc, df)
  dns_enc_out <- transform.autoenc_denoise_e(dns_enc_fit, df)

  dns_dec <- autoenc_denoise_ed(input_size = 5L, encoding_size = 2L, noise_factor = 0.1, num_epochs = 2L)
  dns_dec_fit <- fit.autoenc_denoise_ed(dns_dec, df)
  dns_dec_out <- transform.autoenc_denoise_ed(dns_dec_fit, df)

  vae_enc <- autoenc_variational_e(
    input_size = 5L,
    encoding_size = 2L,
    num_epochs = 2L,
    validation_strategy = "dynamic",
    stopping_rule = "patience"
  )
  vae_enc_fit <- fit.autoenc_variational_e(vae_enc, df)
  vae_enc_out <- transform.autoenc_variational_e(vae_enc_fit, df)

  vae_dec <- autoenc_variational_ed(input_size = 5L, encoding_size = 2L, num_epochs = 2L)
  vae_dec_fit <- fit.autoenc_variational_ed(vae_dec, df)
  vae_dec_out <- transform.autoenc_variational_ed(vae_dec_fit, df)

  expect_equal(dim(dns_enc_out), c(nrow(df), 2L))
  expect_equal(dim(dns_dec_out), c(nrow(df), ncol(df)))
  expect_equal(dim(vae_enc_out), c(nrow(df), 4L))
  expect_equal(dim(vae_dec_out), c(nrow(df), ncol(df)))
})

test_that("stacked and adversarial autoencoders fit and transform through the R wrapper", {
  skip_if_no_pytorch()

  df <- make_autoenc_data(n = 12L)

  sae_enc <- autoenc_stacked_e(input_size = 5L, encoding_size = 2L, k = 2L, num_epochs = 1L)
  sae_enc_fit <- fit.autoenc_stacked_e(sae_enc, df)
  sae_enc_out <- transform.autoenc_stacked_e(sae_enc_fit, df)

  sae_dec <- autoenc_stacked_ed(input_size = 5L, encoding_size = 2L, k = 2L, num_epochs = 1L)
  sae_dec_fit <- fit.autoenc_stacked_ed(sae_dec, df)
  sae_dec_out <- transform.autoenc_stacked_ed(sae_dec_fit, df)

  adv_enc <- autoenc_adv_e(input_size = 5L, encoding_size = 2L, num_epochs = 1L)
  adv_enc_fit <- fit.autoenc_adv_e(adv_enc, df)
  adv_enc_out <- transform.autoenc_adv_e(adv_enc_fit, df)

  adv_dec <- autoenc_adv_ed(input_size = 5L, encoding_size = 2L, num_epochs = 1L)
  adv_dec_fit <- fit.autoenc_adv_ed(adv_dec, df)
  adv_dec_out <- transform.autoenc_adv_ed(adv_dec_fit, df)

  expect_equal(dim(sae_enc_out), c(nrow(df), 2L))
  expect_equal(dim(sae_dec_out), c(nrow(df), ncol(df)))
  expect_equal(dim(adv_enc_out), c(nrow(df), 2L))
  expect_equal(dim(adv_dec_out), c(nrow(df), ncol(df)))
})
