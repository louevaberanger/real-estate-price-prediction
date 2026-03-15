# src/app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("models/price_model.pkl")

# Récupérer les colonnes utilisées par le modèle
columns = model.feature_names_in_

st.title("🏠 Estimation du prix immobilier")

# Champs de saisie pour les caractéristiques principales
surface = st.number_input("Surface reelle bati (m²)", min_value=1, value=100)
pieces = st.number_input("Nombre de pièces principales", min_value=1, value=3)

# Détecter les types locaux présents dans le modèle
types_local = [c.replace("Type local_", "") for c in columns if "Type local_" in c]
type_local = st.selectbox("Type de bien", types_local)

# Détecter les départements présents dans le modèle
departements = [c.replace("Code departement_", "") for c in columns if "Code departement_" in c]
departement = st.selectbox("Code département", departements)

# Valeurs par défaut pour autres colonnes
mois = st.number_input("Mois de mutation", min_value=1, max_value=12, value=1)
annee = st.number_input("Année de mutation", min_value=2000, max_value=2024, value=2024)
surface_terrain = st.number_input("Surface terrain (m²)", min_value=0, value=0)

if st.button("Estimer le prix"):
    # Construire le DataFrame pour la prédiction
    data = pd.DataFrame([{col: 0 for col in columns}])
    data["Surface reelle bati"] = surface
    data["Nombre pieces principales"] = pieces
    data["Surface terrain"] = surface_terrain
    data["mois_mutation"] = mois
    data["annee_mutation"] = annee

    # Remplir les colonnes catégorielles selon le choix
    type_col = f"Type local_{type_local}"
    if type_col in data.columns:
        data[type_col] = 1

    dept_col = f"Code departement_{departement}"
    if dept_col in data.columns:
        data[dept_col] = 1

    # Prédiction (modèle entraîné avec log)
    log_pred = model.predict(data)
    prix_estime = np.exp(log_pred[0])

    st.success(f"💰 Prix estimé : {prix_estime:,.0f} €")