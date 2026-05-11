# Autoencoders — Examples

This section is for readers who want to understand how `daltoolboxdp` brings Python-backed deep learning models into the `daltoolbox` workflow. All examples work on time-series windows, so the main decision is whether you only need a latent representation or whether reconstruction quality is also part of the task.

Use these examples to answer questions such as:

- Do I just need a compact latent representation? Start with the encoder-only block.
- Do I need reconstruction quality as well as encoding? Jump to the encoder-decoder block.
- Do I want temporal memory, convolutional locality, denoising robustness, adversarial regularization, stacking, or a probabilistic latent space? Choose the corresponding family below.

New architecture flexibility
- Dense autoencoders now expose encoder/decoder hidden sizes, hidden activations, decoder output activations, and LeakyReLU slope when applicable.
- Variational autoencoders now expose encoder/decoder depth, reconstruction loss (`"bce"` or `"mse"`), and activation/output activation choices.
- Adversarial autoencoders now expose encoder/decoder/discriminator topologies, dropout, latent prior scale, and optimizer-specific learning rates.
- Stacked autoencoders now allow stage-specific latent sizes and stage-specific hidden layouts, not only a repeated `k`.
- LSTM autoencoders now separate latent size from recurrent hidden size and expose `sequence_length`, `num_layers`, and recurrent dropout.

Encoder only
- [01_autoenc_e.md](01_autoenc_e.md) — Simple autoencoder (encode) for compact latent representations when reconstruction is not the final product.
- [02_autoenc_denoise_e.md](02_autoenc_denoise_e.md) — Denoising autoencoder (encode) for robust latent features under noisy inputs.
- [03_autoenc_conv_e.md](03_autoenc_conv_e.md) — 1D convolutional autoencoder (encode) for local temporal structures within each window.
- [04_autoenc_lstm_e.md](04_autoenc_lstm_e.md) — LSTM autoencoder (encode) for temporal dependencies across ordered steps in the window.
- [05_autoenc_stacked_e.md](05_autoenc_stacked_e.md) — Stacked autoencoder (encode) for gradual nonlinear compression through multiple stages.
- [06_autoenc_variational_e.md](06_autoenc_variational_e.md) — Variational autoencoder (encode) for probabilistic latent representations with KL regularization.
- [07_autoenc_adv_e.md](07_autoenc_adv_e.md) — Adversarial autoencoder (encode) for latent spaces regularized against a chosen prior.

Encoder-decoder
- [11_autoenc_ed.md](11_autoenc_ed.md) — Simple autoencoder (encode-decode) when reconstruction error itself is part of the analysis.
- [12_autoenc_denoise_ed.md](12_autoenc_denoise_ed.md) — Denoising autoencoder (encode-decode) for clean reconstruction from corrupted inputs.
- [13_autoenc_conv_ed.md](13_autoenc_conv_ed.md) — 1D convolutional autoencoder (encode-decode) to reconstruct windows while preserving local patterns.
- [14_autoenc_lstm_ed.md](14_autoenc_lstm_ed.md) — LSTM autoencoder (encode-decode) to reconstruct sequential windows after recurrent compression.
- [15_autoenc_stacked_ed.md](15_autoenc_stacked_ed.md) — Stacked autoencoder (encode-decode) for deeper reconstruction pipelines.
- [16_autoenc_variational_ed.md](16_autoenc_variational_ed.md) — Variational autoencoder (encode-decode) optimizing the ELBO for smooth latent spaces and reconstruction.
- [17_autoenc_adv_ed.md](17_autoenc_adv_ed.md) — Adversarial autoencoder (encode-decode) combining reconstruction with adversarial latent regularization.

Training regimes for the simple encoder-decoder
- [21_autoenc_ed_static_patience.md](21_autoenc_ed_static_patience.md) — Simple autoencoder with static validation and patience-based early stopping.
- [22_autoenc_ed_dynamic_patience.md](22_autoenc_ed_dynamic_patience.md) — Simple autoencoder with dynamic validation and patience-based early stopping.
