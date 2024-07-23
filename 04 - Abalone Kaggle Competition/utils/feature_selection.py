import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_log_error

def rfe_kfold(X, y, model,folds=5, fold_random_state= 1):

    def avaliar_modelo(model, X_train, y_train, X_val, y_val):
        """
        model: modelo já ajustado aos dados de treino
        """
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_score = root_mean_squared_log_error(y_train, y_pred_train)
        val_score = root_mean_squared_log_error(y_val, y_pred_val)

        return train_score, val_score
 
    kf = KFold(n_splits= folds, shuffle= True, random_state= fold_random_state)

    # Quantidade de Features Iniciais
    n_features = X.shape[1]

    # Armazenar os Scores para o dataframe final
    scores = {
        'iteracao' : [],
        'score_treino' : [],
        'score_validacao' : [],
        'feature_list' : [],
        'pior_feature' : [],
        'qtd_features_restantes' : []
    }

    for iteracao in range(n_features):

        #-- Inicializações
        # Pega a lista de features atuais
        feature_list = X.columns.to_list()

        # Inicializa um vetor de 0s com o mesmo tamanho da quantidade de features de X
        importances = np.zeros_like(X.columns)

        # Inicializa as variáveis que irão armazenar os scores
        mean_train_score = 0
        mean_val_score = 0

        for train_idx, val_idx in kf.split(X= X, y= y):

            # Instanciar o modelo
            model = model

            # Filtrando o set de treino
            X_train_cv = X.iloc[train_idx].copy()
            y_train_cv = y.iloc[train_idx].copy()

            # Filtrando o set de validação
            X_val_cv = X.iloc[val_idx].copy()
            y_val_cv = y.iloc[val_idx].copy()

            # Ajustando o modelo
            model = model.fit(X_train_cv, y_train_cv)

            # Calcula as métricas
            train_score, val_score = avaliar_modelo(
                model= model,
                X_train= X_train_cv,
                y_train= y_train_cv,
                X_val= X_val_cv,
                y_val= y_val_cv
            )

            # Salva a média dos scores de treino
            mean_train_score += train_score/folds

            # Salva a média dos scores de validação
            mean_val_score += val_score/folds

            # Salvando a média das feature importances
            importances += model.feature_importances_/folds
        
        # Encontra a pior valor de feature importance
        worst_importance = importances.min()

        # Encontra as posições no array onde temos o pior valor de feature importance
        mask = importances == worst_importance

        # Seleciona as features com pior importance
        worst_features = model.feature_names_in_[mask]

        # Remove as features do modelo
        X = X.drop(columns = worst_features.tolist())

        # Salvando os dados da iteração
        scores['iteracao'].append(iteracao)
        scores['feature_list'].append(feature_list)
        scores['score_treino'].append(mean_train_score)
        scores['score_validacao'].append(mean_val_score)
        scores['pior_feature'].append(worst_features)
        scores['qtd_features_restantes'].append(len(X.columns))

        # Critério de parada: Caso só restem 2 ou menos features
        if X.shape[1] <= 2:
            break

    # Retorna o dataframe com o estudo
    return pd.DataFrame(scores)


def remove_highly_correlated_features(df, threshold):
  # Calculate the correlation matrix
  corr_matrix = df.select_dtypes('number').corr().abs()

  # Select the upper triangle of the correlation matrix
  upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

  # Identify columns to drop based on the threshold
  to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

  # Drop the columns
  df_reduced = df.drop(columns=to_drop)

  return df_reduced, to_drop