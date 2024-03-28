# Importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# Carregando os arquivos CSV
def carregar_dados():
    dfs = []
    for i in range(1, 10):
        df = pd.read_csv(f'data/USC00{i}.csv')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Selecionando colunas do df e preenchendo valores ausentes
def preparar_dados(dados):
    dados_selecionados = dados[['TAVG', 'TMAX', 'TMIN', 'PRCP']]
    return dados_selecionados.fillna(dados_selecionados.mean())

# Criando uma variável indicadora binária
def criar_variavel_alvo(dados):
    return dados['PRCP'].apply(lambda x: 1 if x > 25 else 0)

# Dividindo os dados em conjunto de treinamento e teste
def dividir_dados(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Realizando subamostragem (undersampling)
def realizar_subamostragem(X, y):
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    return undersampler.fit_resample(X, y)

# Realizando sobreamostragem (oversampling)
def realizar_sobreamostragem(X, y):
    oversampler = RandomOverSampler(sampling_strategy='minority')
    return oversampler.fit_resample(X, y)

# Avaliando o desempenho do modelo
def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('')
    print("Acurácia:", acuracia)
    print("Precisão:", precisao)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print('')
    print("Relatório de Classificação:")
    print('')
    print(classification_report(y_test, y_pred))
