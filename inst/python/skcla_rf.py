from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def skcla_rf_create(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                  max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                  n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None,
                  ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=int(verbose),
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
        monotonic_cst=monotonic_cst
    )
    return model

def skcla_rf_train(model, df_train, target_column):
    df_train = pd.DataFrame(df_train)  # garante consistência com R
    #print("Column types:", df_train.dtypes)
    #print("Shape of data:", df_train.shape)

    X_train = df_train.drop(columns=[target_column])  # preserva nomes das colunas
    y_train = df_train[target_column].values

    model.fit(X_train, y_train)
    return model

def skcla_rf_predict(model, df_test):
    try:
        df_test = pd.DataFrame(df_test)  # garante estrutura consistente
        #print(df_test)
        predictions = model.predict(df_test)  # mantém colunas nomeadas
        return predictions.tolist()  # para compatibilidade com R
    except TypeError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

def skcla_rf_fit(model, df_train, target_column):
    return skcla_rf_train(model, df_train, target_column)
