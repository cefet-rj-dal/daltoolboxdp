from imblearn.over_sampling import SMOTE

def inbalanced_create_model(random_state=42):
    smote = SMOTE(random_state=int(random_state))
    return smote

def fit_resample(select_method, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    X_train_smote, y_train_smote = select_method.fit_resample(X_train, y_train)
    
    return  X_train_smote, y_train_smote
