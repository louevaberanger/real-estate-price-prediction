import pandas as pd
import numpy as np
import joblib

# Charger les modèles
model_standard = joblib.load("models/price_model_lgbm_standard.pkl")
model_luxury = joblib.load("models/price_model_lgbm_luxury.pkl")

# Charger les colonnes exactes utilisées à l'entraînement
columns_standard = joblib.load("models/columns_standard.pkl")
columns_luxury = joblib.load("models/columns_luxury.pkl")

# Charger le mapping code_postal -> prix_m2
prix_m2_dep_dict = joblib.load("models/prix_m2_dep.pkl")  

def predict_price(surface, pieces, terrain, type_local, code_departement):
    """
    Prédit le prix d'un bien immobilier selon ses caractéristiques.
    Switch automatique vers modèle luxe si prix > 1M€.
    """

    # Récupérer le prix moyen au m² du code postal
    prix_m2_dep = prix_m2_dep_dict.get(code_departement, np.mean(list(prix_m2_dep_dict.values())))

    # 1️⃣ Construire dict avec toutes les colonnes numériques et dérivées
    data = {
        "Surface reelle bati": surface,
        "Nombre pieces principales": pieces,
        "Surface terrain": terrain,
        "prix_m2_dep": prix_m2_dep,
        "surface_log": np.log1p(surface),
        "terrain_log": np.log1p(terrain),
        "surface_par_piece": surface / pieces if pieces > 0 else 0,
        "surface_x_prix_dep": surface * prix_m2_dep,
        "terrain_ratio": terrain / surface if surface > 0 else 0,
        "surface_totale": surface + terrain,
        "ratio_bati_terrain": surface / terrain if terrain > 0 else 0,
        "luxury_flag": int(surface > 200 or terrain > 500)
    }

    # 2️⃣ Ajouter les colonnes catégorielles si elles existent
    type_col = f"Type local_{type_local}"
    dept_col = f"Code departement_{code_departement}"
    if type_col in columns_standard:
        data[type_col] = 1
    if dept_col in columns_standard:
        data[dept_col] = 1

    # 3️⃣ DataFrame pour modèle standard
    df_standard = pd.DataFrame([data]).reindex(columns=columns_standard, fill_value=0)

    # 4️⃣ Prédiction standard
    log_pred_standard = model_standard.predict(df_standard)
    prix_estime = np.expm1(log_pred_standard[0])

    # 5️⃣ Si >1M€, passer au modèle luxe
    if prix_estime > 1_000_000:
        if type_col in columns_luxury:
            data[type_col] = 1
        if dept_col in columns_luxury:
            data[dept_col] = 1
        df_luxury = pd.DataFrame([data]).reindex(columns=columns_luxury, fill_value=0)
        log_pred_luxury = model_luxury.predict(df_luxury)
        prix_estime = np.expm1(log_pred_luxury[0])

    return prix_estime

# Exemple d'utilisation pour Paris 75001
prix = predict_price(
    surface=50,
    pieces=2,
    terrain=300,
    type_local="Maison",
    code_departement="92"
)
print("Prix estimé :", prix)
