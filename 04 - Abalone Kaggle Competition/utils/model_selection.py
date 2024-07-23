import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

def RMSLE_by_model(model_name, model, X_train, y_train, X_test, y_test):

    y_train_pred = abs(model.predict(X_train))
    y_test_pred = abs(model.predict(X_test))

    RMSLE_train = root_mean_squared_log_error(y_train, y_train_pred)
    RMSLE_test = root_mean_squared_log_error(y_test, y_test_pred)

    metrics_df = pd.DataFrame({
        'Modelo' : [model_name, model_name],
        'Set' : ['Treino', 'Teste'],
        'RMSLE' : [RMSLE_train, RMSLE_test]
    })

    return metrics_df