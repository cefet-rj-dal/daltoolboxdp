from sklearn.naive_bayes import GaussianNB
import numpy as np

def nb_create(priors=None, var_smoothing=1e-9):
    model = GaussianNB(
        priors=priors,
        var_smoothing=var_smoothing
    )
    return model

def nb_fit(model, df_train, target_column):
    try:
        X_train = df_train.drop(target_column, axis=1).values
        y_train = df_train[target_column].values
        
        # Print debugging info
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_train data type: {X_train.dtype}")
        print(f"y_train data type: {y_train.dtype}")
        
        # Check for NaN values
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("Warning: NaN values detected in training data")
            # Replace NaNs with 0 to avoid training errors
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
        
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in nb_fit: {str(e)}")
        # Return the unfitted model rather than None
        return model

def nb_predict(model, df_test):
    try:
        # Convert dataframe to numpy array
        if hasattr(df_test, 'values'):
            X_test = df_test.values
        else:
            X_test = np.array(df_test)
        
        # Print debugging info
        print(f"X_test shape: {X_test.shape}")
        print(f"X_test data type: {X_test.dtype}")
        
        # Check for NaN values
        if np.isnan(X_test).any():
            print("Warning: NaN values detected in test data")
            X_test = np.nan_to_num(X_test)
        
        predictions = model.predict(X_test)
        return predictions
    except TypeError as e:
        print(f"TypeError in nb_predict: {e}")
        # Return empty array instead of None
        return np.array([])
    except Exception as e:
        print(f"Error in nb_predict: {e}")
        # Return empty array instead of None
        return np.array([])
