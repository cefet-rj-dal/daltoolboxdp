"""
SMOTE oversampling wrapper used by daltoolboxdp via reticulate.

Intended R entry points (names on the R side may differ slightly):
  - inbalanced_create_model(random_state=42) -> SMOTE instance
  - fit_resample(select_method, df_train, target_column) -> (X_resampled, y_resampled)

Data expectations:
  - df_train: pandas.DataFrame with features + target column.
  - target_column: name of the target column to rebalance.
  - Returns NumPy arrays for easy consumption on the R side.
"""

from imblearn.over_sampling import SMOTE


def inbalanced_create_model(random_state=42):
    """Create a SMOTE oversampler with a fixed random_state for reproducibility."""
    smote = SMOTE(random_state=int(random_state))
    return smote


def fit_resample(select_method, df_train, target_column):
    """Apply SMOTE to df_train and return resampled feature and label arrays.

    Parameters
    ----------
    select_method : SMOTE
        A configured SMOTE object created by inbalanced_create_model().
    df_train : pandas.DataFrame
        Training data including the target column.
    target_column : str
        Name of the target column.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Resampled feature matrix and label vector.
    """
    # Expect DataFrame; keep numpy arrays for scikit-learn compatibility
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    X_train_smote, y_train_smote = select_method.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote
