# Autoencoders — Examples

Collection of autoencoder examples applied to time-series windows. Each link points to the corresponding `.md` file and includes a brief description.

- [autoenc_e.md](autoenc_e.md) — Simple autoencoder (encode) that learns a latent representation (p->k) by minimizing reconstruction loss.
- [autoenc_ed.md](autoenc_ed.md) — Simple autoencoder (encode-decode) trained end-to-end to reconstruct (p<->k) and evaluate reconstruction error.
- [autoenc_adv_e.md](autoenc_adv_e.md) — Adversarial autoencoder (encode) that regularizes the latent space via a discriminator to match a prior.
- [autoenc_adv_ed.md](autoenc_adv_ed.md) — Adversarial autoencoder (encode-decode) combining reconstruction loss with an adversarial game in latent space.
- [autoenc_conv_e.md](autoenc_conv_e.md) — 1D convolutional autoencoder (encode) to capture local patterns within windows.
- [autoenc_conv_ed.md](autoenc_conv_ed.md) — 1D convolutional autoencoder (encode-decode) reconstructing from the compressed code.
- [autoenc_denoise_e.md](autoenc_denoise_e.md) — Denoising autoencoder (encode) trained to recover clean inputs from corrupted data.
- [autoenc_denoise_ed.md](autoenc_denoise_ed.md) — Denoising autoencoder (encode-decode) with stochastic input noise and clean reconstruction.
- [autoenc_lstm_e.md](autoenc_lstm_e.md) — LSTM autoencoder (encode) that encodes temporal dependencies into a fixed-size vector.
- [autoenc_lstm_ed.md](autoenc_lstm_ed.md) — LSTM autoencoder (encode-decode) to encode (p->k) and reconstruct (k->p) time-series windows.
- [autoenc_stacked_e.md](autoenc_stacked_e.md) — Stacked autoencoder (encode) with multiple nonlinear layers for gradual compression.
- [autoenc_stacked_ed.md](autoenc_stacked_ed.md) — Stacked autoencoder (encode-decode) with a bottleneck and reconstruction guided by loss.
- [autoenc_variational_e.md](autoenc_variational_e.md) — Variational autoencoder (VAE, encode) with KL regularization for a probabilistic latent space.
- [autoenc_variational_ed.md](autoenc_variational_ed.md) — Variational autoencoder (encode-decode) optimizing the ELBO (reconstruction + KL) for a smooth latent space.
