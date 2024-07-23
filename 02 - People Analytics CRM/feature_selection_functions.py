from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

def amostragem(df,tamanho_amostra):
    amostra = df.sample(n=tamanho_amostra, random_state= 2236)
    return amostra


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


def variancia(table_01, threshold):
    cat_attributes = table_01.select_dtypes(include='object')
    num_attributes = table_01.select_dtypes(exclude='object')

    # Preparação dos dados

    # Create a label encoder
    label_encoder = LabelEncoder()

    # Fit and transform the data
    for obj in cat_attributes.columns:
        cat_attributes[obj] = label_encoder.fit_transform(cat_attributes[obj].astype(str))

    table_final = pd.concat([num_attributes,cat_attributes],axis=1)

    # Scaler
    scaler = StandardScaler()

    table_final_02 = scaler.fit_transform(table_final)

    table_final_03 = pd.DataFrame(table_final_02,columns=table_01.columns)

    selector = VarianceThreshold(threshold)
    selector.fit_transform(table_final_03)

    # Colunas selecionadas
    selected_features = table_final_03.columns[selector.get_support()]

    # Manter apenas features selecionadas
    table_final_03_select01 = table_final_03[selected_features]

    return table_final_03_select01


def generate_metadata(dataframe):
    """
    Gera um dataframe contendo metadados das colunas do dataframe fornecido.

    :param dataframe: DataFrame para o qual os metadados serão gerados.
    :return: DataFrame contendo metadados.
    """

    # Coleta de metadados básicos
    metadata = pd.DataFrame({
        'nome_variavel': dataframe.columns,
        'tipo': dataframe.dtypes,
        'qt_nulos': dataframe.isnull().sum(),
        'percent_nulos': round((dataframe.isnull().sum() / len(dataframe))* 100,2),
        'cardinalidade': dataframe.nunique(),
    })
    metadata=metadata.sort_values(by='percent_nulos',ascending=False)
    metadata = metadata.reset_index(drop=True)

    return metadata


def feature_importance(df):
    # Amostragem
    amostra = df.sample(n=85000, random_state= 2236)

    X = amostra.drop(columns=['SK_ID_CURR','TARGET'])
    y = amostra.TARGET
    id = amostra.SK_ID_CURR

    # Tratamento de missings
    from sklearn.impute import SimpleImputer
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    cat_attributes = X.select_dtypes(include='object')
    num_attributes = X.select_dtypes(exclude='object')
  
    if not cat_attributes.empty:
        cat_imputed = imputer_cat.fit_transform(cat_attributes)
        df_cat = pd.DataFrame(cat_imputed,columns=cat_attributes.columns)
    else:
        df_cat = pd.DataFrame()

    # Aplicação de encoding
    label_encoder = LabelEncoder()

    # Fit and transform the data
    for obj in cat_attributes.columns:
        df_cat[obj] = label_encoder.fit_transform(df_cat[obj].astype(str))
    
    if not num_attributes.empty:
        num_imputed = imputer_num.fit_transform(num_attributes)
        df_num = pd.DataFrame(num_imputed,columns=num_attributes.columns)
    else:
        df_num = pd.DataFrame()

    try:
        y.index = df_num.index
        id.index = df_num.index
        df_tratado = pd.concat([df_num,df_cat,y],axis=1)

        X = df_tratado.drop(columns='TARGET')
        y = df_tratado.TARGET

        # Treino utilizando o Gradient Boosting
        algoritmo = GradientBoostingClassifier(random_state=0)
        algoritmo.fit(X,y)
        feature_importances = algoritmo.feature_importances_

        # Importância das variáveis de acordo com o algoritmo

        df_importancias = pd.DataFrame(X.columns,columns=['Variável'])
        df_importancias['Importância'] = feature_importances

        df_importancias = df_importancias[df_importancias['Importância'] > 0]

        return df_importancias
    except:
        return pd.DataFrame()
    


def vars_selection(df,percentual_preenchimento,threshold,tamanho_amostragem):

    amostra = amostragem(df, tamanho_amostragem)

    metadata_df = generate_metadata(amostra)

    # Avaliar preenchimento
    vars = metadata_df[metadata_df.percent_nulos <= percentual_preenchimento]['nome_variavel']
    # Avaliar correlaçao
    df_reduced, dropped_features = remove_highly_correlated_features(amostra[vars], threshold=threshold)
    # Avaliar variância - remover variáveis constantes
    table_01 = variancia(df_reduced, threshold=0)

    vars_selected = table_01.columns.to_list()
    vars_df_final = vars_selected
    df_selected = df[vars_df_final]

    df_importancias = feature_importance(df_selected)

    return df_importancias