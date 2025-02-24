from collections import Counter
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def inbalanced_create_model(random_state=42):
    stomek = SMOTETomek(random_state=random_state)
    return stomek

def fit_resample(select_method, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    X_train_smote, y_train_smote = select_method.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote