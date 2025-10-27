"""
SVC (Support Vector Classifier) wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_svc.R):
  - skcla_svc_create(...hyperparams...) -> sklearn model
  - skcla_svc_fit(model, df_train, target_column, slevels=None) -> fitted model
  - skcla_svc_predict(model, df_test) -> list of labels
"""

from sklearn.svm import SVC
import pandas as pd

def skcla_svc_create(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, 
               C=1.0, shrinking=True, probability=False, cache_size=200, 
               class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
               break_ties=False, random_state=None):
    
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state
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

def skcla_svc_fit(model, df_train, target_column, slevels=None):
    """Entry from R to fit; delegates to skcla_svc_train."""
    return skcla_svc_train(model, df_train, target_column)
