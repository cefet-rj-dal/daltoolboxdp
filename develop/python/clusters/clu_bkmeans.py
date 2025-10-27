from sklearn.cluster import BisectingKMeans

def clu_bkmeans_create(n_clusters=8, *, init='random', n_init=1, random_state=None, max_iter=300, verbose=0, tol=0.0001, copy_x=True, algorithm='lloyd', bisecting_strategy='biggest_inertia'):
    model = BisectingKMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        copy_x=copy_x,
        algorithm=algorithm,
        bisecting_strategy=bisecting_strategy
    )
    return model

def clu_bkmeans_train(model, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_bkmeans_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_bkmeans_fit_predict(model, df_train):
    """
    Fit the Bisecting KMeans model and predict cluster labels for the training data.
    """
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    predictions = model.fit_predict(X_train)
    return model, predictions

def clu_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_bkmeans_train(model, df_train, target_column)
