# Detecção de Fraudes em Cartões de Crédito

## Sobre o projeto

Este projeto tem como objetivo desenvolver modelos de Machine Learning capazes de identificar transações fraudulentas realizadas com cartões de crédito.

O conjunto de dados utilizado apresenta um grande desbalanceamento entre as classes: existem muito mais transações legítimas do que fraudulentas. Esse cenário representa um desafio para os modelos de classificação, pois um modelo pode apresentar uma boa taxa de acerto geral e, mesmo assim, deixar de identificar grande parte das fraudes.

Por esse motivo, a avaliação prioriza métricas como **Recall, Precision e F1-score da classe de fraude**, permitindo analisar não apenas os acertos gerais, mas também a capacidade de identificar transações fraudulentas.

## Preparação dos dados

Durante a preparação, foram realizadas as seguintes etapas:

- Carregamento do conjunto de dados utilizando Pandas.
- Análise da distribuição das classes e identificação do desbalanceamento.
- Verificação de valores ausentes e registros duplicados.
- Criação da variável `Amount_log`, utilizando transformação logarítmica.
- Padronização da variável `Amount` com StandardScaler.
- Divisão dos dados em conjuntos de treinamento e teste, utilizando `stratify` para preservar a proporção das classes.

## Modelos utilizados

Foram desenvolvidos e avaliados diferentes modelos de classificação:

- Regressão Logística;
- Random Forest;
- XGBoost;
- Regressão Logística com Undersampling;
- Regressão Logística com SMOTE.

Também foram utilizados métodos de balanceamento para investigar como a distribuição das classes influencia o desempenho dos modelos.

O Undersampling reduz a quantidade de exemplos da classe majoritária, enquanto o SMOTE cria exemplos sintéticos da classe minoritária. Essas técnicas foram aplicadas somente aos dados de treinamento, mantendo os dados de teste com sua distribuição original.

## Comparação dos modelos

O desempenho foi avaliado utilizando três métricas principais da classe de fraude:

- **Precision:** indica quantas transações identificadas como fraude realmente eram fraudulentas.
- **Recall:** indica quantas das fraudes reais foram identificadas pelo modelo.
- **F1-score:** representa o equilíbrio entre Precision e Recall.

### Resultados

| Modelo | Precision | Recall | F1-score |
|---|---|---|---|
| Regressão Logística | A preencher | A preencher | A preencher |
| Random Forest | A preencher | A preencher | A preencher |
| XGBoost | A preencher | A preencher | A preencher |
| Logística com Undersampling | A preencher | A preencher | A preencher |
| Logística com SMOTE | A preencher | A preencher | A preencher |

Os resultados permitem comparar o desempenho dos modelos e observar como as técnicas de balanceamento influenciam a identificação de fraudes.

Além disso, foram utilizadas as curvas ROC e Precision-Recall para complementar a avaliação.

## Ajuste do limiar de decisão (Threshold)

Foi realizado um teste com o limiar de decisão de **0.3** na Regressão Logística e no XGBoost.

O objetivo foi investigar como a alteração do threshold influencia a identificação das transações fraudulentas, buscando aumentar o Recall sem desconsiderar a Precision.

A escolha do limiar envolve um equilíbrio entre identificar mais fraudes e evitar que transações legítimas sejam classificadas incorretamente.

## Explicabilidade com SHAP

Foi utilizada a biblioteca SHAP para compreender melhor o comportamento do modelo XGBoost.

Foram geradas visualizações para analisar:

- A importância geral das variáveis nas previsões do modelo.
- A influência das variáveis em uma previsão individual.

Os gráficos permitem observar quais características tiveram maior influência nas decisões do modelo e compreender melhor como as classificações são realizadas.

**Principais variáveis identificadas:** preencher após conferir o gráfico SHAP.

## Alterações e melhorias em relação à versão da Expert

Em relação à estrutura inicial utilizada como referência no desafio, foram realizadas melhorias e adaptações, incluindo:

- Implementação de Undersampling e SMOTE para tratar o desbalanceamento.
- Comparação entre diferentes modelos de classificação.
- Avaliação utilizando Precision, Recall e F1-score.
- Testes com diferentes limiares de decisão.
- Utilização do GridSearchCV para buscar melhores parâmetros para o XGBoost.
- Aplicação do SHAP para interpretação das previsões.
- Ajustes na preparação dos dados para evitar vazamento de informações entre treinamento e teste.

## Tecnologias e bibliotecas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP

## Aprendizagem

O desenvolvimento deste projeto permitiu praticar conceitos de Machine Learning, principalmente classificação, preparação de dados, tratamento de classes desbalanceadas, avaliação de modelos e explicabilidade.

Também foi possível compreender a importância de escolher métricas adequadas ao problema, especialmente em situações nas quais a identificação da classe minoritária é fundamental.

---

**Projeto desenvolvido para fins de estudo e prática em Ciência de Dados e Machine Learning.**
