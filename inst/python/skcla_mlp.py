"""
MLPClassifier wrapper used by daltoolboxdp via reticulate.

R entry points (see R/skcla_mlp.R):
  - skcla_mlp_create(...hyperparams...) -> sklearn model
  - skcla_mlp_fit(model, df_train, target_column) -> fitted model
  - skcla_mlp_predict(model, df_test) -> list of labels
"""

from sklearn.neural_network import MLPClassifier
import pandas as pd

def skcla_mlp_create(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
               random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, 
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
               beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, max_fun=15000):

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
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun
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
