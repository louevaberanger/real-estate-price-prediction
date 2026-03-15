import pandas as pd
import joblib
import numpy as np

# charger le modèle
model = joblib.load("models/price_model.pkl")

# créer un DataFrame avec toutes les colonnes du modèle
columns = model.feature_names_in_  # les colonnes utilisées à l'entraînement

# remplir avec des valeurs par défaut (0) ou celles que tu veux tester
data = pd.DataFrame([{col: 0 for col in columns}])

# remplacer les colonnes connues
data["Surface reelle bati"] = 100
data["Nombre pieces principales"] = 4
data["Type local_Maison"] = 1          # si tu veux simuler une maison
data["Code departement_28"] = 1        # par exemple Paris

# Prédiction en log
log_pred = model.predict(data)

# Transformation inverse
prix_estime = np.exp(log_pred[0])
print("Prix estimé (€) :", prix_estime)
