# Importando bibliotecas
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import uniform
from utils import carregar_dados, preparar_dados, criar_variavel_alvo, dividir_dados, realizar_subamostragem, realizar_sobreamostragem, avaliar_modelo


# Carregando e preparando os dados
dados = carregar_dados()
X = preparar_dados(dados)
y = criar_variavel_alvo(dados)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = dividir_dados(X, y)

# Verificando balanceamento antes da subamostragem e sobreamostragem
print("Balanceamento antes da subamostragem:")
print(y_train.value_counts())

# Realizando subamostragem
X_under, y_under = realizar_subamostragem(X_train, y_train)

# Verificando balanceamento após a subamostragem
print("\nBalanceamento após a subamostragem:")
print(pd.Series(y_under).value_counts())

# Realizando sobreamostragem
X_over, y_over = realizar_sobreamostragem(X_train, y_train)

# Verificando balanceamento após a sobreamostragem
print("\nBalanceamento após a sobreamostragem:")
print(pd.Series(y_over).value_counts())

# Criando pipeline do modelo LogisticRegression
pipeline_regressao_logistica = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Treinando modelo de Regressão Logística
print("\nTreinando modelo de Regressão Logística...")
pipeline_regressao_logistica.fit(X_train, y_train)
print("Avaliação do modelo de Regressão Logística:")
avaliar_modelo(pipeline_regressao_logistica, X_test, y_test)

# Criando pipeline do modelo DecisionTreeClassifier
pipeline_arvore_decisao = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Treinando modelo de Árvore de Decisão
print("\nTreinando modelo de Árvore de Decisão...")
pipeline_arvore_decisao.fit(X_train, y_train)
print("Avaliação do modelo de Árvore de Decisão sem GridSearchCV:")
avaliar_modelo(pipeline_arvore_decisao, X_test, y_test)

# Definindo os parâmetros para a busca exaustiva
parametros_arvore_decisao = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': range(1, 11)
}

# Criando o pipeline da Árvore de Decisão com GridSearchCV
pipeline_arvore_decisao = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Inicializando o GridSearchCV
grid_search_arvore_decisao = GridSearchCV(
    estimator=pipeline_arvore_decisao,
    param_grid=parametros_arvore_decisao,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Treinando o modelo com GridSearchCV
grid_search_arvore_decisao.fit(X_train, y_train)

# Avaliando o melhor modelo da Árvore de Decisão
melhor_modelo_arvore_decisao = grid_search_arvore_decisao.best_estimator_
print("Melhores Parâmetros para a Árvore de Decisão com GridSearchCV:", grid_search_arvore_decisao.best_params_)
print("Avaliação do melhor modelo da Árvore de Decisão com GridSearchCV:")
avaliar_modelo(melhor_modelo_arvore_decisao, X_test, y_test)

# Criando o pipeline do modelo SVM
pipeline_svm = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', SVC(random_state=42))
])

# Treinando modelo de SVM
print("\nTreinando modelo de SVM...")
pipeline_svm.fit(X_train, y_train)
print("Avaliação do modelo de SVM:")
avaliar_modelo(pipeline_svm, X_test, y_test)

# Definindo os parâmetros para a busca aleatória
parametros_svm = {
    'classifier__C': uniform(0.1, 10),
    'classifier__kernel': ['linear', 'poly', 'rbf']
}

# Criando o pipeline do SVM com RandomizedSearchCV
pipeline_svm_random = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', SVC(random_state=42))
])

# Inicializando o RandomizedSearchCV
randomized_search_svm = RandomizedSearchCV(
    estimator=pipeline_svm_random,
    param_distributions=parametros_svm,
    scoring='accuracy',
    cv=5,
    n_iter=10,
    random_state=42,
    n_jobs=-1
)

# Treinando o modelo com RandomizedSearchCV
randomized_search_svm.fit(X_train, y_train)

# Avaliando o melhor modelo SVM
melhor_modelo_svm = randomized_search_svm.best_estimator_
print("Melhores Parâmetros para SVM com RandomizedSearchCV:", randomized_search_svm.best_params_)
print("Avaliação do melhor modelo SVM:")
avaliar_modelo(melhor_modelo_svm, X_test, y_test)