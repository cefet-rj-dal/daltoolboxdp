from sklearn.neighbors import KNeighborsRegressor

def cla_knn_create(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'):
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric
    )
    return model

def cla_knn_train(model, df_train, target_column):
    print("Tipos das colunas:", df_train.dtypes)
    print("Shape dos dados:", df_train.values.shape)
    X_train = df_train.drop(target_column, axis=1).values

    y_train = df_train[target_column].values
    model.fit(X_train, y_train)
    return model

def cla_knn_predict(model, df_test):
    try:
        predictions = model.predict(df_test.values)
        return predictions
    except TypeError as e:
        print(f"Erro encontrado: {e}")
    except Exception as e:
        print(f"Outro erro ocorreu: {e}")

  
def cla_knn_fit(model, df_train, target_column, n_epochs=None, lr=None):
    return cla_knn_train(model, df_train, target_column)
