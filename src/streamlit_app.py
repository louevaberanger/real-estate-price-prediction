import streamlit as st
import pandas as pd
import numpy as np
import joblib
from predict import predict_price  # ta fonction ci-dessus
model_standard = joblib.load("models/price_model_lgbm_standard.pkl")
model_luxury = joblib.load("models/price_model_lgbm_luxury.pkl")
columns_standard = joblib.load("models/columns_standard.pkl")
columns_luxury = joblib.load("models/columns_luxury.pkl")

st.title("🏠 Estimation du prix immobilier")

# ---------------------------
# 1️⃣ Inputs utilisateur
# ---------------------------
# Charger la liste des communes et types depuis ton jeu de colonnes
departement = [c.replace("Code departement_", "") for c in columns_standard if "Code departement_" in c]
types_local = [c.replace("Type local_", "") for c in columns_standard if "Type local_" in c]


surface = st.number_input("Surface bâtie (m²)", min_value=1, max_value=5000, value=100)
pieces = st.number_input("Nombre de pièces principales", min_value=1, max_value=15, value=3)
surface_terrain = st.number_input("Surface terrain (m²)", min_value=0, max_value=10000, value=0)

type_local = st.selectbox("Type de bien", types_local)
departement = st.selectbox("Departement", departement)
# ---------------------------
# 2️⃣ Bouton prédiction
# ---------------------------
if st.button("Estimer le prix"):
    prix_estime = predict_price(
        surface=surface,
        pieces=pieces,
        terrain=surface_terrain,
        type_local=type_local,
        code_departement=departement
    )
    st.success(f"💰 Prix estimé : {prix_estime:,.0f} €")