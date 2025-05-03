from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def create_rf_model(n_estimators=100, random_state=0, X=None, y=None):
    model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state))
    model.fit(X, y)
    return model

def fs_create(model, threshold="mean", prefit=True):
    sf_method = SelectFromModel(estimator=model, threshold=threshold, prefit=prefit)
    return sf_method

def fit(select_method, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    select_method.fit(X_train, y_train)
    return select_method

def fit_transform(select_method, df_train, target_column):
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    X_train = select_method.fit_transform(X_train, y_train)
    return X_train
