from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def create_lg_model(C=0.1, penalty='l1', solver='liblinear', X=None, y=None):
    model = LogisticRegression(C=C, penalty=penalty, solver=solver)
    model.fit(X, y)
    return model

def fs_create(model, threshold="mean", prefit=True):
    sf_method = SelectFromModel(estimator=model, threshold=threshold, prefit=prefit)
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
    y_train = df_train[target_column].values
    X_train = select_method.fit_transform(X_train, y_train)
    return X_train
