from sklearn.mixture import GaussianMixture

def clu_gmm_create(n_components=1, *, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
        weights_init=weights_init,
        means_init=means_init,
        precisions_init=precisions_init,
        random_state=random_state,
        warm_start=warm_start,
        verbose=verbose,
        verbose_interval=verbose_interval
    )
    return model

def clu_gmm_train(model, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_gmm_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Erro encontrado: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_gmm_fit_predict(model, df_train):
    """
    Fit the Bisecting KMeans model and predict cluster labels for the training data.
    """
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    predictions = model.fit_predict(X_train)
    return model, predictions

def clu_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_gmm_train(model, df_train, target_column)
