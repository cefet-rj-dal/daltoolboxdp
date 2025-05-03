from sklearn.svm import SVC


def svc_create(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, 
               C=1.0, shrinking=True, probability=False, cache_size=200, 
               class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
               break_ties=False, random_state=None):
    
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state
    )
    
    return model

def svc_train(model, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def svc_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error encountered: {e}")
    except Exception as e:
        print(f"Another error occurred: {e}")

def svc_fit(model, df_train, target_column, slevels=None):
    return svc_train(model, df_train, target_column)
