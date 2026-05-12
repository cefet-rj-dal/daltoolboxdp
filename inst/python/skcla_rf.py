"""
RandomForestClassifier wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_rf.R):
  - skcla_rf_create(...hyperparams...) -> sklearn model
  - skcla_rf_fit(model, df_train, target_column) -> fitted model
  - skcla_rf_predict(model, df_test) -> list of labels (for R compatibility)
  - skcla_rf_predict_proba(model, df_test) -> list of per-class probabilities

Data expectations: pandas.DataFrame; target_column is present in df_train and excluded for prediction.
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def skcla_rf_create(n_estimators=100, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='sqrt', class_weight=None):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        class_weight=class_weight,
    )
    return model

def skcla_rf_train(model, df_train, target_column):
    """Fit RandomForestClassifier. df_train must include the target_column."""
    df_train = pd.DataFrame(df_train)  # garante consistência com R
    #print("Column types:", df_train.dtypes)
    #print("Shape of data:", df_train.shape)

    X_train = df_train.drop(columns=[target_column])  # preserva nomes das colunas
    y_train = df_train[target_column].values

    model.fit(X_train, y_train)
    return model

def skcla_rf_predict(model, df_test):
    """Predict labels as a Python list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)  # garante estrutura consistente
        #print(df_test)
        predictions = model.predict(df_test)  # mantém colunas nomeadas
        return predictions.tolist()  # para compatibilidade com R
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

def skcla_rf_predict_proba(model, df_test):
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

def skcla_rf_fit(model, df_train, target_column):
    """Entry from R to fit; delegates to skcla_rf_train."""
    return skcla_rf_train(model, df_train, target_column)
