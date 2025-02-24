from sklearn.ensemble import GradientBoostingClassifier

def cla_gb_create(loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                  min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, 
                  max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
    
    model = GradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_depth=max_depth,
        min_impurity_decrease=min_impurity_decrease,
        init=init,
        random_state=random_state,
        max_features=max_features,
        verbose=verbose,
        max_leaf_nodes=max_leaf_nodes,
        warm_start=warm_start,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        ccp_alpha=ccp_alpha
    )
    
    return model

def cla_gb_train(model, df_train, target_column):
    print("Column types:", df_train.dtypes)
    print("Data shape:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values
    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def cla_gb_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Error encountered: {e}")
    except Exception as e:
        print(f"Another error occurred: {e}")

def cla_gb_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return cla_gb_train(model, df_train, target_column)
