"""
SMOTE + TomekLinks combined resampling wrapper used by daltoolboxdp via reticulate.

Intended R entry points:
  - inbalanced_create_model(random_state=42) -> SMOTETomek instance
  - fit_resample(select_method, df_train, target_column) -> (X_resampled, y_resampled)

Data expectations:
  - df_train: pandas.DataFrame with features + target column.
  - target_column: name of the target column to rebalance.
  - Returns NumPy arrays for easy consumption on the R side.
"""

from imblearn.combine import SMOTETomek


def inbalanced_create_model(random_state=42):
    """Create a SMOTETomek resampler with a fixed random_state for reproducibility."""
    stomek = SMOTETomek(random_state=int(random_state))
    return stomek


def fit_resample(select_method, df_train, target_column):
    """Apply SMOTETomek to df_train and return resampled feature and label arrays.

    Parameters
    ----------
    select_method : SMOTETomek
        A configured resampler created by inbalanced_create_model().
    df_train : pandas.DataFrame
        Training data including the target column.
    target_column : str
        Name of the target column.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Resampled feature matrix and label vector.
    """
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    X_res, y_res = select_method.fit_resample(X_train, y_train)
    return X_res, y_res
