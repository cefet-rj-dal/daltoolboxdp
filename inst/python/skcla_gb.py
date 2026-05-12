"""
GradientBoostingClassifier wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_gb.R):
  - skcla_gb_create(...hyperparams...) -> sklearn model
  - skcla_gb_fit(model, df_train, target_column) -> fitted model
  - skcla_gb_predict(model, df_test) -> list of labels
  - skcla_gb_predict_proba(model, df_test) -> list of per-class probabilities
"""

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

def skcla_gb_create(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
                  min_samples_split=2, min_samples_leaf=1, loss='log_loss'):
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        loss=loss,
    )
    
    return model

def skcla_gb_train(model, df_train, target_column):
    """Fit GradientBoostingClassifier. df_train must include the target_column."""
    df_train = pd.DataFrame(df_train)
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.shape)

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column].values

    model.fit(X_train, y_train)
    return model

def skcla_gb_predict(model, df_test):
    """Predict labels as a Python list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)
        #print(df_test)
        predictions = model.predict(df_test)
        return predictions.tolist()
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def skcla_gb_predict_proba(model, df_test):
    """Predict class probabilities as a nested list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)
        probabilities = model.predict_proba(df_test)
        return probabilities.tolist()
    except TypeError as e:
        print(f"Error occurred: {e}")
        return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def skcla_gb_fit(model, df_train, target_column, n_epochs=None, lr=None):
    """Entry from R to fit; delegates to skcla_gb_train."""
    return skcla_gb_train(model, df_train, target_column)
