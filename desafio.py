
# ==========================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap


# ==========================================
# 2. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ==========================================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# Verificar a proporção das classes
print("Distribuição das classes:")
print(df["Class"].value_counts())

print("\nProporção das classes:")
print(df["Class"].value_counts(normalize=True))

# Criar variável com transformação logarítmica
df["Amount_log"] = np.log1p(df["Amount"])

print("\nValores de Amount e Amount_log:")
print(df[["Amount", "Amount_log"]].head())

# Informações gerais
df.info()

# Verificar valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum().sum())

# Verificar registros duplicados
print("\nRegistros duplicados:")
print(df.duplicated().sum())

# Gráfico de distribuição das transações
sns.countplot(data=df, x="Class", color="green")
plt.title("Distribuição das transações")
plt.xlabel("Classe (0 = Normal, 1 = Fraude)")
plt.ylabel("Quantidade")
plt.show()


# ==========================================
# 3. PREPARAÇÃO DOS DADOS
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.3,
    random_state=42
)

# Padronizar Amount usando apenas os dados de treino
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train["Amount_scaled"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount_scaled"] = scaler.transform(
    X_test[["Amount"]]
)


# ==========================================
# 4. REGRESSÃO LOGÍSTICA
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nResultado da Regressão Logística:")
print(classification_report(y_test, y_pred))


# ==========================================
# 5. CURVA ROC E AUC
# ==========================================

y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))


# ==========================================
# 6. CURVA PRECISION-RECALL
# ==========================================

precision, recall, _ = precision_recall_curve(
    y_test,
    y_probs
)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# ==========================================
# 7. UNDERSAMPLING
# ==========================================

# Juntar temporariamente as variáveis de treino com a classe
dados_treino = X_train.copy()
dados_treino["Class"] = y_train

# Separar as transações normais e fraudulentas
fraudes = dados_treino[dados_treino["Class"] == 1]
normais = dados_treino[dados_treino["Class"] == 0]

# Reduzir a quantidade de transações normais
normais_under = normais.sample(
    n=len(fraudes),
    random_state=42
)

# Juntar as duas classes
df_under = pd.concat([fraudes, normais_under])

# Embaralhar os dados
df_under = df_under.sample(frac=1, random_state=42)

# Separar variáveis e classe
X_train_under = df_under.drop("Class", axis=1)
y_train_under = df_under["Class"]

# Conferir o equilíbrio
print("\nAntes do undersampling:")
print(y_train.value_counts())

print("\nDepois do undersampling:")
print(y_train_under.value_counts())

# Treinar modelo com undersampling
model_under = LogisticRegression(max_iter=1000)

model_under.fit(X_train_under, y_train_under)

# Avaliar nos dados de teste originais
y_pred_under = model_under.predict(X_test)

print("\nResultado com Undersampling:")
print(classification_report(y_test, y_pred_under))


# ==========================================
# 8. OVERSAMPLING COM SMOTE
# ==========================================

smote = SMOTE(random_state=42)

# Aplicar SMOTE somente nos dados de treinamento
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# Conferir o equilíbrio
print("\nDepois do SMOTE:")
print(y_train_smote.value_counts())

# Treinar modelo com SMOTE
model_smote = LogisticRegression(max_iter=1000)

model_smote.fit(X_train_smote, y_train_smote)

# Avaliar nos dados de teste originais
y_pred_smote = model_smote.predict(X_test)

print("\nResultado com SMOTE:")
print(classification_report(y_test, y_pred_smote))


# ==========================================
# 9. RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nResultado do Random Forest:")
print(classification_report(y_test, y_pred_rf))


# ==========================================
# 10. AJUSTE DE THRESHOLD DA REGRESSÃO LOGÍSTICA
# ==========================================

threshold = 0.3

y_pred_custom = (y_probs >= threshold).astype(int)

print("\nRegressão Logística com threshold 0.3:")
print(classification_report(y_test, y_pred_custom))


# ==========================================
# 11. XGBOOST
# ==========================================

xgb = XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\nResultado do XGBoost:")
print(classification_report(y_test, y_pred_xgb))


# ==========================================
# 12. AJUSTE DE THRESHOLD DO XGBOOST
# ==========================================

xgb_probs = xgb.predict_proba(X_test)[:, 1]

threshold_xgb = 0.3

y_pred_xgb_custom = (
    xgb_probs >= threshold_xgb
).astype(int)

print("\nXGBoost com threshold 0.3:")
print(classification_report(y_test, y_pred_xgb_custom))


# ==========================================
# 13. IMPORTÂNCIA DAS VARIÁVEIS
# ==========================================

importancias = xgb.feature_importances_

plt.bar(range(len(importancias)), importancias)
plt.title("Importância das variáveis")
plt.xlabel("Variáveis")
plt.ylabel("Importância")
plt.show()


# ==========================================
# 14. GRIDSEARCHCV
# ==========================================

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=10
    ),
    param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nMelhores parâmetros:")
print(grid.best_params_)

print("Melhor Recall:")
print(grid.best_score_)


# ==========================================
# 15. EXPLICABILIDADE COM SHAP
# ==========================================

explainer = shap.Explainer(xgb)

shap_values = explainer(X_test.iloc[:100])

# Importância geral das variáveis
shap.plots.bar(shap_values)

# Explicação de uma previsão individual
shap.plots.waterfall(shap_values[0])


# ==========================================
# 16. COMPARAÇÃO DOS MODELOS
# ==========================================

# Previsões dos modelos
pred_log = model.predict(X_test)
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)
pred_under = model_under.predict(X_test)
pred_smote = model_smote.predict(X_test)

resultados = {
    "Modelo": [
        "Regressão Logística",
        "Random Forest",
        "XGBoost",
        "Logística com Undersampling",
        "Logística com SMOTE"
    ],

    "Precision": [
        precision_score(y_test, pred_log),
        precision_score(y_test, pred_rf),
        precision_score(y_test, pred_xgb),
        precision_score(y_test, pred_under),
        precision_score(y_test, pred_smote)
    ],

    "Recall": [
        recall_score(y_test, pred_log),
        recall_score(y_test, pred_rf),
        recall_score(y_test, pred_xgb),
        recall_score(y_test, pred_under),
        recall_score(y_test, pred_smote)
    ],

    "F1-score": [
        f1_score(y_test, pred_log),
        f1_score(y_test, pred_rf),
        f1_score(y_test, pred_xgb),
        f1_score(y_test, pred_under),
        f1_score(y_test, pred_smote)
    ]
}

comparacao = pd.DataFrame(resultados)

print("\nComparação dos modelos:")
print(comparacao)
