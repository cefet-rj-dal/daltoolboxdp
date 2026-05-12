"""
KNeighborsClassifier wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_knn.R):
  - skcla_knn_create(...hyperparams...) -> sklearn model
  - skcla_knn_fit(model, df_train, target_column) -> fitted model
  - skcla_knn_predict(model, df_test) -> list of labels
  - skcla_knn_predict_proba(model, df_test) -> list of per-class probabilities
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def skcla_knn_create(n_neighbors=5, weights='uniform', metric='euclidean'):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    return model

def skcla_knn_fit(model, df_train, target_column):
    """Fit KNN. df_train must include the target_column."""
    try:
        df_train = pd.DataFrame(df_train)

        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column].values

        #print(f"X_train shape: {X_train.shape}")
        #print(f"y_train shape: {y_train.shape}")

        if X_train.isnull().values.any() or pd.isnull(y_train).any():
            #print("Warning: NaN values detected in training data")
            X_train = X_train.fillna(0)
            y_train = np.nan_to_num(y_train)

        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in skcla_knn_fit: {str(e)}")
        return model

def skcla_knn_predict(model, df_test):
    """Predict labels as a Python list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)
        #print(f"X_test shape: {df_test.shape}")

        if df_test.isnull().values.any():
            #print("Warning: NaN values detected in test data")
            df_test = df_test.fillna(0)

        predictions = model.predict(df_test)
        return predictions.tolist()
    except TypeError as e:
        print(f"TypeError in skcla_knn_predict: {e}")
        return []
    except Exception as e:
        print(f"Error in skcla_knn_predict: {e}")
        return []

def skcla_knn_predict_proba(model, df_test):
    """Predict class probabilities as a nested list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)

        if df_test.isnull().values.any():
            df_test = df_test.fillna(0)

        probabilities = model.predict_proba(df_test)
        return probabilities.tolist()
    except TypeError as e:
        print(f"TypeError in skcla_knn_predict_proba: {e}")
        return []
    except Exception as e:
        print(f"Error in skcla_knn_predict_proba: {e}")
        return []
