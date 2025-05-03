from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

def skcla_nb_create(priors=None, var_smoothing=1e-9):
    model = GaussianNB(
        priors=priors,
        var_smoothing=var_smoothing
    )
    return model

def skcla_nb_fit(model, df_train, target_column):
    try:
        df_train = pd.DataFrame(df_train)
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column].values

        #print(f"X_train shape: {X_train.shape}")
        #print(f"y_train shape: {y_train.shape}")
        #print(f"X_train data type: {X_train.dtypes}")
        #print(f"y_train data type: {y_train.dtype}")

        # Check and replace NaNs
        if X_train.isnull().values.any() or pd.isnull(y_train).any():
            #print("Warning: NaN values detected in training data")
            X_train = X_train.fillna(0)
            y_train = np.nan_to_num(y_train)

        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in skcla_nb_fit: {str(e)}")
        return model

def skcla_nb_predict(model, df_test):
    try:
        df_test = pd.DataFrame(df_test)

        print(f"X_test shape: {df_test.shape}")
        print(f"X_test data type: {df_test.dtypes}")

        if df_test.isnull().values.any():
            #print("Warning: NaN values detected in test data")
            df_test = df_test.fillna(0)

        predictions = model.predict(df_test)
        return predictions.tolist()
    except TypeError as e:
        print(f"TypeError in skcla_nb_predict: {e}")
        return []
    except Exception as e:
        print(f"Error in skcla_nb_predict: {e}")
        return []
