from cmath import inf
from sklearn.cluster import OPTICS

def clu_optics_create(*, min_samples=5, max_eps=inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None, n_jobs=None):
    model = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        p=p,
        metric_params=metric_params,
        cluster_method=cluster_method,
        eps=eps,
        xi=xi,
        predecessor_correction=predecessor_correction,
        algorithm=algorithm,
        leaf_size=leaf_size,
        memory=memory,
        min_cluster_size=min_cluster_size,
        n_jobs=n_jobs
    )
    return model

def clu_optics_train(model, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_optics_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_optics_fit_predict(model, df_train):
    """
    Fit the Bisecting KMeans model and predict cluster labels for the training data.
    """
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    predictions = model.fit_predict(X_train)
    return model, predictions

def clu_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_optics_train(model, df_train, target_column)
