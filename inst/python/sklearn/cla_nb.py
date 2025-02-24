from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

def nb_create(priors=None, var_smoothing=1e-9):
    model = GaussianNB(
        priors=priors,
        var_smoothing=var_smoothing
    )
    return model

def nb_fit(model, df_train, target_column):
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def nb_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error encountered: {e}")
    except Exception as e:
        print(f"Another error occurred: {e}")
