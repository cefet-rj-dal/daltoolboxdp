from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def fs_create(n_features_to_select=0.5, lg_max_iter=1000):
    model = LogisticRegression(max_iter=int(lg_max_iter))
    sf_method = RFE(estimator=model, n_features_to_select=n_features_to_select)
    return sf_method

def fit(select_method, df_train, target_column):
    #print("Column types:", df_train.dtypes)
    #print("Data shape:", df_train.shape)  # Use .shape instead of values.shape
    X_train = df_train.drop(columns=[target_column]).values
    y_train = df_train[target_column].values
    select_method.fit(X_train, y_train)
    return select_method

def fit_transform(select_method, df_train, target_column):
    X_train = df_train.drop(columns=[target_column]).values
    y_train = df_train[target_column].values
    X_train_selected = select_method.fit_transform(X_train, y_train)

    # Convert to NumPy array (just for safety)
    X_train_selected = np.array(X_train_selected)

    # Convert to Pandas DataFrame with appropriate column names
    selected_features = df_train.drop(columns=[target_column]).columns[select_method.get_support()]
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)

    #print("Final returned object type:", type(X_train_selected_df))

    return X_train_selected_df  # This is now a Pandas DataFrame

