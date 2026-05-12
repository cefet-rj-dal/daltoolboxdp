"""
SVC (Support Vector Classifier) wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_svc.R):
  - skcla_svc_create(...hyperparams...) -> sklearn model
  - skcla_svc_fit(model, df_train, target_column, slevels=None) -> fitted model
  - skcla_svc_predict(model, df_test) -> list of labels
  - skcla_svc_predict_proba(model, df_test) -> list of per-class probabilities
"""

from sklearn.svm import SVC
import pandas as pd

def skcla_svc_create(C=1.0, kernel='rbf', gamma='scale', degree=3, coef0=0.0, probability=False, class_weight=None):
    
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        probability=probability,
        class_weight=class_weight
    )
    
    return model

def skcla_svc_train(model, df_train, target_column):
    """Fit SVC. df_train must include the target_column."""
    df_train = pd.DataFrame(df_train)

    #print("Column types:")
    #print(df_train.dtypes)
    #print("Data shape:", df_train.shape)

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column].values

    model.fit(X_train, y_train)
    return model

def skcla_svc_predict(model, df_test):
    """Predict labels as a Python list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)
        #print("Prediction input shape:", df_test.shape)

        predictions = model.predict(df_test)
        return predictions.tolist()
    except TypeError as e:
        print(f"Error occurred: {e}")
        return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def skcla_svc_predict_proba(model, df_test):
    """Predict class probabilities when available; otherwise return an empty list."""
    try:
        df_test = pd.DataFrame(df_test)
        if not hasattr(model, "predict_proba"):
            return []
        probabilities = model.predict_proba(df_test)
        return probabilities.tolist()
    except TypeError as e:
        print(f"Error occurred: {e}")
        return []
    except Exception as e:
        if "predict_proba" not in str(e):
            print(f"Error occurred: {e}")
        return []

def skcla_svc_fit(model, df_train, target_column, slevels=None):
    """Entry from R to fit; delegates to skcla_svc_train."""
    return skcla_svc_train(model, df_train, target_column)
