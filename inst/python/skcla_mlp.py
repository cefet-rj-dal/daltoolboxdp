"""
MLPClassifier wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_mlp.R):
  - skcla_mlp_create(...hyperparams...) -> sklearn model
  - skcla_mlp_fit(model, df_train, target_column) -> fitted model
  - skcla_mlp_predict(model, df_test) -> list of labels
  - skcla_mlp_predict_proba(model, df_test) -> list of per-class probabilities
"""

from sklearn.neural_network import MLPClassifier
import pandas as pd

def skcla_mlp_create(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
               learning_rate_init=0.001, max_iter=200, early_stopping=False):

    if activation is None:
        activation = 'relu'  
    
    if solver is None:
        solver = 'adam'
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
    )
    return model

def skcla_mlp_fit(model, df_train, target_column):
    """Fit MLP. df_train must include the target_column."""
    df_train = pd.DataFrame(df_train)
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column].values

    model.fit(X_train, y_train)
    return model

def skcla_mlp_predict(model, df_test):
    """Predict labels as a Python list to simplify R interop."""
    try:
        df_test = pd.DataFrame(df_test)
        predictions = model.predict(df_test)
        return predictions.tolist()
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

def skcla_mlp_predict_proba(model, df_test):
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
