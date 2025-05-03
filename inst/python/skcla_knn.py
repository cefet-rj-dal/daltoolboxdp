from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def skcla_knn_create(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, 
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

def skcla_knn_fit(model, df_train, target_column):
    try:
        df_train = pd.DataFrame(df_train)

        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column].values

        #print(f"X_train shape: {X_train.shape}")
        #print(f"y_train shape: {y_train.shape}")

        if np.isnan(X_train).values.any() or np.isnan(y_train).any():
            #print("Warning: NaN values detected in training data")
            X_train = X_train.fillna(0)
            y_train = np.nan_to_num(y_train)

        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error in skcla_knn_fit: {str(e)}")
        return model

def skcla_knn_predict(model, df_test):
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
