from sklearn.cluster import KMeans

def clu_kmeans_create(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd'):
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm
    )
    return model

def clu_kmeans_train(model, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_kmeans_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Erro encontrado: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_kmeans_fit_predict(model, df_train):
    """
    Fit the KMeans model and predict cluster labels for the training data.
    """
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    predictions = model.fit_predict(X_train)
    return model, predictions

def clu_kmeans_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_kmeans_train(model, df_train, target_column)
