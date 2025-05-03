from sklearn.feature_selection import VarianceThreshold

def fs_create(threshold=0.2):
    sf_method = VarianceThreshold(threshold=float(threshold))
    return sf_method

def fit(select_method, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    select_method.fit(X_train, y_train)
    return select_method

def fit_transform(select_method, df_train, target_column):
    X_train = df_train.drop(target_column, axis=1).values
    X_train = select_method.fit_transform(X_train)
    return X_train
