from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_create(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, 
               p=2, metric='minkowski', metric_params=None, n_jobs=None):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs
    )
    return model

def knn_fit(model, df_train, target_column):
    try:
        X_train = df_train.drop(target_column, axis=1).values
        y_train = df_train[target_column].values
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("Warning: NaN values detected in training data")
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
        
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in knn_fit: {str(e)}")
        return model

def knn_predict(model, df_test):
    try:
        if hasattr(df_test, 'values'):
            X_test = df_test.values
        else:
            X_test = np.array(df_test)
        
        print(f"X_test shape: {X_test.shape}")
        
        if np.isnan(X_test).any():
            print("Warning: NaN values detected in test data")
            X_test = np.nan_to_num(X_test)
        
        predictions = model.predict(X_test)
        return predictions
    except TypeError as e:
        print(f"TypeError in knn_predict: {e}")
        return np.array([])
    except Exception as e:
        print(f"Error in knn_predict: {e}")
        return np.array([])
