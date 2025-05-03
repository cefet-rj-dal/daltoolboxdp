from sklearn.cluster import AgglomerativeClustering

def clu_agg_create(n_clusters=2, *, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        memory=memory,
        connectivity=connectivity,
        compute_full_tree=compute_full_tree,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances
    )
    return model

def clu_agg_train(model, df_train):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.values
    model.fit(X_train)
    return model

def clu_agg_predict(model, df_test):
    try:
        if hasattr(model, 'fit_predict'):
            predictions = model.fit_predict(df_test.values)
            return predictions
        else:
            raise NotImplementedError("AgglomerativeClustering does not support prediction on new data.")
    except Exception as e:
        print(f"An error occurred: {e}")