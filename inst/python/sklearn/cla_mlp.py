from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

def mlp_create(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
               learning_rate='constant', max_iter=200, tol=0.0001, random_state=None):
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    return model

def mlp_fit(model, df_train, target_column):
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def mlp_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error encountered: {e}")
    except Exception as e:
        print(f"Another error occurred: {e}")
