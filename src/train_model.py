import pandas as pd
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processing import clean_data
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Charger et nettoyer les données
# ---------------------------
df = pd.read_csv(
    "H:/Desktop/data_real_estate/dvf_2024.csv",
    sep="|",
    low_memory=False
)
df = clean_data(df)
print("Nombre de lignes après nettoyage :", len(df))

# ---------------------------
# 2️⃣ Target log-transformée
# ---------------------------
y = df["Valeur_fonciere_log"]
X = df.drop(["Valeur_fonciere_log"], axis=1)

# ---------------------------
# 3️⃣ Split train/test
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Séparer biens chers et normaux
seuil_luxury = 1_000_000
mask_train_luxury = np.exp(y_train) > seuil_luxury
mask_train_standard = ~mask_train_luxury

X_train_standard = X_train[mask_train_standard]
y_train_standard = y_train[mask_train_standard]

X_train_luxury = X_train[mask_train_luxury]
y_train_luxury = y_train[mask_train_luxury]

X_train_luxury = pd.concat([X_train_luxury]*3, ignore_index=True)
y_train_luxury = pd.concat([y_train_luxury]*3, ignore_index=True)


# Paramètres LightGBM

params_standard = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

params_luxury = params_standard.copy()
params_luxury['num_leaves'] = 63
params_luxury['max_depth'] = 10

# ---------------------------
# 6️⃣ Dataset LightGBM
# ---------------------------
train_data_standard = lgb.Dataset(X_train_standard, label=y_train_standard)
train_data_luxury = lgb.Dataset(X_train_luxury, label=y_train_luxury)

# Colonnes pour le modèle standard
columns_standard = X_train_standard.columns.tolist()
joblib.dump(columns_standard, "models/columns_standard.pkl")

# Colonnes pour le modèle luxe
columns_luxury = X_train_luxury.columns.tolist()
joblib.dump(columns_luxury, "models/columns_luxury.pkl")

# ---------------------------
# 7️⃣ Entraînement modèle principal
# ---------------------------
model_standard = lgb.train(
    params_standard,
    train_data_standard,
    num_boost_round=1000,
    valid_sets=[train_data_standard],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# ---------------------------
# 8️⃣ Entraînement modèle luxe
# ---------------------------
model_luxury = lgb.train(
    params_luxury,
    train_data_luxury,
    num_boost_round=1000,
    valid_sets=[train_data_luxury],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# ---------------------------
# 9️⃣ Prédictions combinées sur test set
# ---------------------------
mask_test_luxury = np.exp(y_test) > seuil_luxury
mask_test_standard = ~mask_test_luxury

preds = np.zeros(len(y_test))
preds[mask_test_standard] = np.expm1(model_standard.predict(X_test[mask_test_standard]))
preds[mask_test_luxury] = np.expm1(model_luxury.predict(X_test[mask_test_luxury]))

y_test_euros = np.expm1(y_test)

# ---------------------------
# 10️⃣ Évaluation
# ---------------------------
mae_global = mean_absolute_error(y_test_euros, preds)
mae_luxury = mean_absolute_error(y_test_euros[mask_test_luxury], preds[mask_test_luxury])
mape_luxury = np.mean(np.abs((y_test_euros[mask_test_luxury] - preds[mask_test_luxury]) / y_test_euros[mask_test_luxury])) * 100

rmse = np.sqrt(mean_squared_error(y_test_euros, preds))
r2 = r2_score(y_test_euros, preds)

print(f"MAE global : {mae_global:.2f} €")
print(f"MAPE biens chers : {mape_luxury:.2f}%")
print(f"MAE biens chers : {mae_luxury:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R² : {r2:.3f}")

# ---------------------------
# 11️⃣ Visualisation
# ---------------------------
plt.scatter(y_test_euros, preds, alpha=0.3)
plt.xlabel("Prix réel (€)")
plt.ylabel("Prix prédit (€)")
plt.title("Comparaison prix réel vs prédit")
plt.show()

# ---------------------------
# 12️⃣ Sauvegarde des modèles
# ---------------------------
joblib.dump(model_standard, "models/price_model_lgbm_standard.pkl")
joblib.dump(model_luxury, "models/price_model_lgbm_luxury.pkl")


