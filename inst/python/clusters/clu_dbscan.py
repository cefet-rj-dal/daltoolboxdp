from sklearn.cluster import DBSCAN

def clu_dbscan_create(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
    model = DBSCAN(
        eps = eps,
        min_samples=min_samples,
        metric = metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs
    )
    return model

def clu_dbscan_train(model, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_dbscan_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Erro encontrado: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_dbscan_train(model, df_train, target_column)
