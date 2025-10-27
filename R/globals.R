# globals.R

# Declara símbolos externos resolvidos em tempo de execução (ex.: entradas Python
# carregadas via `reticulate`) para suprimir NOTES de globais indefinidos no R CMD check.

utils::globalVariables(c(
  "ts_lstm_create", "ts_lstm_fit", "ts_lstm_predict",
  "ts_conv1d_create","ts_conv1d_fit", "ts_conv1d_predict",
  "autoenc_create", "autoenc_fit", "autoenc_encode", "autoenc_encode_decode",
  "autoenc_adv_create", "autoenc_adv_fit", "autoenc_adv_encode", "autoenc_adv_encode_decode",
  "autoenc_conv_create", "autoenc_conv_fit", "autoenc_conv_encode", "autoenc_conv_encode_decode",
  "autoenc_lstm_create", "autoenc_lstm_fit", "autoenc_lstm_encode", "autoenc_lstm_encode_decode",
  "autoenc_stacked_create", "autoenc_stacked_fit", "autoenc_stacked_encode", "autoenc_stacked_encode_decode",
  "autoenc_variational_create", "autoenc_variational_fit", "autoenc_variational_encode", "autoenc_variational_encode_decode",
  "autoenc_denoise_create", "autoenc_denoise_fit", "autoenc_denoise_encode", "autoenc_denoise_encode_decode",
  "skcla_gb_create", "skcla_gb_fit", "skcla_gb_predict",
  "skcla_knn_create", "skcla_knn_fit", "skcla_knn_predict",
  "skcla_rf_create", "skcla_rf_fit", "skcla_rf_predict",
  "skcla_mlp_create", "skcla_mlp_fit", "skcla_mlp_predict",
  "skcla_nb_create", "skcla_nb_fit", "skcla_nb_predict",
  "skcla_svc_create", "skcla_svc_fit", "skcla_svc_predict",
  # recém-adicionados para suprimir NOTES
  "create_lg_model", "create_rf_model", "fs_create", "fit_transform", "inbalanced_create_model",
  # IMB
  "create_imb_smote_model", "fit_imb_resample_smote",
  "create_imb_smotetomek_model", "fit_imb_resample_smotetomek",
  "create_imb_tomek_model", "fit_imb_resample_tomek",
  # FE
  "create_fe_variance_threshold_model", "fit_transform_fe_variance_threshold",
  "create_fe_selectkbest_model", "fit_transform_fe_selectkbest",
  "create_fe_rf_model", "create_fe_selectfrommodel_rf", "fit_transform_fe_selectfrommodel_rf",
  "create_fe_lg_model", "create_fe_selectfrommodel_lg", "fit_transform_fe_selectfrommodel_lg",
  "create_fe_smote_model", "fit_transform_fe_smote",
  "create_fe_sequential_fs_model", "fit_transform_fe_sequential_fs",
  "create_fe_rfe_model", "fit_transform_fe_rfe"
))
