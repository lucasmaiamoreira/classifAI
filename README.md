# Projeto: Previsão de Fenômenos Climáticos

## Este repositório contém um código em Python para um projeto de análise de dados climáticos e construção de modelos de aprendizado de máquina para prever a ocorrência de chuva em determinadas regiões.

## Descrição do Projeto
O projeto consiste nas seguintes etapas:

1. Carregamento dos Dados: Os dados climáticos são carregados a partir de arquivos CSV.

2. Preparação dos Dados: As colunas relevantes são selecionadas e os valores ausentes são preenchidos com a média dos dados.

3. Criação da Variável Alvo: Uma variável indicadora binária é criada com base no volume de precipitação.

4. Divisão dos Dados: Os dados são divididos em conjuntos de treinamento e teste.

5. Balanceamento dos Dados: São realizadas técnicas de subamostragem e sobreamostragem para lidar com desequilíbrios nos dados.

6. Treinamento e Avaliação de Modelos: São construídos modelos de Regressão Logística, Árvore de Decisão e SVM. Os modelos são treinados e avaliados usando métricas de desempenho como acurácia, precisão, recall e F1-Score.

7. Tuning de Hiperparâmetros: Para os modelos de Árvore de Decisão e SVM, são realizadas buscas de hiperparâmetros usando GridSearchCV e RandomizedSearchCV, respectivamente.

## Como Usar
Pré-requisitos: Certifique-se de ter o Python instalado, juntamente com as bibliotecas Pandas, NumPy, Scikit-learn e Matplotlib.

1. Clonando o Repositório: Clone este repositório em seu ambiente local.

2. Configurar um Ambiente Virtual (Opcional, mas Recomendado).

3. Instale as Dependências.

4. Executando o Código: Execute o arquivo Python main.py para carregar os dados, treinar os modelos e visualizar os resultados.

Explorando os Resultados: Analise as métricas de desempenho e gráficos gerados para entender a eficácia dos modelos.

## Estrutura do Código
- main.py: Contém o código principal para execução do projeto.

- utils.py: Contém funções auxiliares utilizadas no projeto.
- data/: Pasta que contém os arquivos CSV de dados climáticos.
- README.md: Este arquivo, fornecendo informações sobre o projeto e instruções para utilização.

## Autor
Este projeto foi desenvolvido por Lucas Maia Moreira.

## Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença
Este projeto está licenciado sob a Licença MIT.