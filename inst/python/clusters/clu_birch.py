from sklearn.cluster import Birch

def clu_birch_create(*, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy='deprecated'):
    model = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
        compute_labels=compute_labels,
        copy=copy
    )

def clu_birch_train(model, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def clu_birch_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Erro encontrado: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

def clu_birch_fit_predict(model, df_train):
    """
    Fit the Bisecting KMeans model and predict cluster labels for the training data.
    """
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    predictions = model.fit_predict(X_train)
    return model, predictions

def clu_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return clu_birch_train(model, df_train, target_column)
