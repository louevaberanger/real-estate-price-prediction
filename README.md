# real-estate-price-prediction

Real Estate Price Prediction

Projet de prédiction du prix des biens immobiliers en France à partir des données DVF (Demande de Valeurs Foncières).

📂 Structure du projet

real-estate-price-prediction/
│
├─ data/                 # Données brutes et traitées
│  └─ raw/               # CSV DVF original (non commité sur GitHub à cause de la taille)
│
├─ models/               # Modèles entraînés sauvegardés (.pkl)
│
├─ src/                  # Scripts Python
│  ├─ data_processing.py # Fonctions pour nettoyer et transformer les données
│  ├─ train_model.py     # Entraînement du modèle
│  ├─ predict.py         # Faire des prédictions sur de nouvelles données
│
├─ notebooks/            # Exploration et tests en Jupyter Notebook
│
├─ requirements.txt      # Librairies nécessaires
└─ README.md             # Ce fichier

⚡ Installation

Cloner le projet :

git clone https://github.com/louevaberanger/real-estate-price-prediction.git
cd real-estate-price-prediction

Créer un environnement virtuel (optionnel mais recommandé) :

python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

Installer les dépendances :

pip install -r requirements.txt

🧹 Nettoyage des données

Le script data_processing.py contient la fonction clean_data(df) qui :

Transforme les valeurs numériques (Valeur fonciere) en float.

Supprime les lignes sans informations essentielles (surface, valeur foncière, code postal).

Filtre les prix et surfaces extrêmes.

Remplit les valeurs manquantes (Nombre pieces principales, Surface terrain).

Transforme les dates en année et mois.

Encode les variables catégorielles (Type local, Code departement) pour le machine learning.

Création de nouvelles features :

- surface_log : transformation logarithmique de la surface
- terrain_log : transformation logarithmique du terrain
- surface_par_piece : surface moyenne par pièce
- prix_moyen_cp : prix moyen observé dans le même code postal

🏗️ Entraînement du modèle

Exemple avec train_model.py :

python src/train_model.py

Le modèle est un XGBoost (plus performant que Random Forest qui a été testé en premier) avec :

- n_estimators ≈ 1000
- learning_rate ≈ 0.02
- max_depth ≈ 7
- subsample ≈ 0.8

La target (Valeur fonciere) est log-transformée pour réduire l’impact des valeurs extrêmes.

Le modèle entraîné est sauvegardé dans models/price_model.pkl.

Évaluation

Le script renvoie :

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R²

Ces métriques permettent de mesurer la qualité des prédictions.

## 📈 Résultats du modèle

MAE : 77 097 €

RMSE : 131 824 €

R² : 0.563

Le modèle explique environ **56 % de la variance du prix des biens immobiliers**.

🧮 Faire une prédiction

Exemple avec predict.py :

python src/predict.py

Charger le modèle : joblib.load("models/price_model.pkl")

Créer un DataFrame avec les colonnes attendues (features du modèle)

Faire la prédiction (log-transform inversée si nécessaire)

Exemple d’utilisation :

import pandas as pd
import joblib
import numpy as np

model = joblib.load("models/price_model.pkl")

data = pd.DataFrame({
    "Surface reelle bati": [100],
    "Nombre pieces principales": [4],
    "Type local_Maison": [1],
    "Code departement_75": [1],
    # ajouter toutes les colonnes nécessaires
})

log_pred = model.predict(data)
prix_estime = np.exp(log_pred[0])
print("Prix estimé (€) :", prix_estime)

📊 Analyse des variables

Le modèle permet d’extraire l’importance des features :

import pandas as pd

importances = pd.DataFrame({
    "feature": model.feature_names_in_,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importances.head(10))

Cela montre quelles variables influencent le plus le prix (ex. Surface reelle bati, Nombre pieces principales, Code departement_75, etc.)

🌐 Interface utilisateur (optionnelle)

streamlit run src/app.py

Streamlit : interface web pour saisir les caractéristiques et afficher le prix estimé.


🔧 Remarques

Les données DVF sont très volumineuses, donc l’entraînement peut être long.

Le modèle est plus précis sur les biens standards (prix < 1,5M €).

Les DOM-TOM sont inclus avec les codes départements > 97.

Les colonnes catégorielles doivent correspondre exactement à celles utilisées à l’entraînement pour éviter les erreurs de prédiction.

Le modèle entraîné est sauvegardé localement dans models/price_model.pkl.

Ce fichier n'est pas versionné sur GitHub car il est trop volumineux.
Il peut être recréé en exécutant :

python src/train_model.py

## 🚀 Améliorations possibles

- Ajouter des coordonnées géographiques (latitude / longitude)
- Utiliser des modèles spatiaux
- Ajouter des données socio-économiques par commune
- Déployer l'application via Streamlit Cloud

![Model prediction](prediction_plot.png)
