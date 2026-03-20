import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import xgboost

from data_processing import clean_data

# Charger les données
df = pd.read_csv(
    "H:/Desktop/data_real_estate/dvf_2024.csv",
    sep="|",
    low_memory=False
)

# Nettoyer les données
df = clean_data(df)
print("Nombre de lignes après nettoyage :", len(df))

# Transforme la target en log pour stabiliser les valeurs extrêmes
df["log_valeur_fonciere"] = np.log(df["Valeur fonciere"])

# Séparer features et target
X = df.drop(["Valeur fonciere", "log_valeur_fonciere"], axis=1)
y = df["log_valeur_fonciere"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Créer et entraîner le modèle
from xgboost import XGBRegressor

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Prédictions
log_preds = model.predict(X_test)

# Retour à l'échelle euros
preds = np.exp(log_preds)
y_test_euros = np.exp(y_test)

mae = mean_absolute_error(y_test_euros, preds)
mse = mean_squared_error(y_test_euros, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_euros, preds)

print(f"MAE : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R² : {r2:.3f}")

import matplotlib.pyplot as plt

plt.scatter(y_test_euros, preds, alpha=0.3)
plt.xlabel("Prix réel (€)")
plt.ylabel("Prix prédit (€)")
plt.title("Comparaison prix réel vs prédit")
plt.show()

# Sauvegarder le modèle
joblib.dump(model, "models/price_model.pkl")
