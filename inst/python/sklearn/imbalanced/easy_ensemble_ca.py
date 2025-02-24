from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def inbalanced_create_model(random_state=42, n_estimators=10):
    easy = EasyEnsembleClassifier(random_state=random_state, n_estimators=n_estimators)
    return easy

def fit(select_method, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    easy_ensemble = select_method.fit(X_train, y_train)
    return easy_ensemble