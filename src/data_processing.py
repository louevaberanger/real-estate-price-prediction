import pandas as pd
import numpy as np
import joblib

def clean_data(df):
    # ------------------------
    # 1️⃣ Nettoyage basique
    # ------------------------
    df["Valeur fonciere"] = df["Valeur fonciere"].str.replace(",", ".")
    df["Valeur fonciere"] = pd.to_numeric(df["Valeur fonciere"], errors="coerce")
    
    df["Code postal"] = df["Code postal"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Supprimer lignes sans valeurs essentielles
    df = df[df["Valeur fonciere"].notna()]
    df = df[df["Surface reelle bati"].notna() & (df["Surface reelle bati"] > 0)]
    df = df[(df["Valeur fonciere"] > 10000) & (df["Valeur fonciere"] < 10000000)]

    # ------------------------
    # 2️⃣ Prix au m²
    # ------------------------
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    prix_dep = df.groupby("Code departement")["prix_m2"].mean()
    df["prix_m2_dep"] = df["Code departement"].map(prix_dep)

    df = df[(df["prix_m2"] > 500) & (df["prix_m2"] < 20000)]
    df = df[df["Surface reelle bati"] < 1000]

    # ------------------------
    # 3️⃣ Remplissage NaN
    # ------------------------
    df["Nombre pieces principales"].fillna(df["Nombre pieces principales"].median(), inplace=True)
    df["Surface terrain"].fillna(0, inplace=True)

    # ------------------------
    # 4️⃣ Date mutation
    # ------------------------
    df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")
    df["annee_mutation"] = df["Date mutation"].dt.year
    df["mois_mutation"] = df["Date mutation"].dt.month

    # ------------------------
    # 5️⃣ Code département
    # ------------------------
    df["Code departement"] = df["Code postal"].str.zfill(5).str[:2]
    
    # Créer le mapping prix_m2 par département
    prix_m2_dep = df.groupby("Code departement")["prix_m2"].mean().to_dict()
    joblib.dump(prix_m2_dep, "models/prix_m2_dep.pkl")

    # ------------------------
    # 6️⃣ Encodage catégoriel
    # ------------------------
    df = pd.get_dummies(df, columns=["Type local", "Code departement"], drop_first=False)

    
    # Features supplémentaires
   
    df["Valeur_fonciere_log"] = np.log1p(df["Valeur fonciere"])
    df["surface_log"] = np.log1p(df["Surface reelle bati"])
    df["terrain_log"] = np.log1p(df["Surface terrain"])
    df["surface_par_piece"] = df["Surface reelle bati"] / df["Nombre pieces principales"].replace(0,1)
    df["surface_x_prix_dep"] = df["Surface reelle bati"] * df["prix_m2_dep"]
    df["terrain_ratio"] = df["Surface terrain"] / df["Surface reelle bati"].replace(0,1)
    
   
    # Features supplémentaires pour biens chers
   
    # Somme des surfaces des lots Carrez (pour biens avec plusieurs lots)
    carrez_cols = [
    "Surface Carrez du 1er lot",
    "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot",
    "Surface Carrez du 4eme lot",
    "Surface Carrez du 5eme lot"
    ]

    # garder seulement celles présentes
    carrez_cols = [c for c in carrez_cols if c in df.columns]

    if len(carrez_cols) > 0:

        # conversion en float
        for col in carrez_cols:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["surface_lots_total"] = df[carrez_cols].fillna(0).sum(axis=1)

    else:
        df["surface_lots_total"] = 0
    
    # Nombre de lots
    df["nb_lots"] = df["Nombre de lots"].fillna(1)

    # Surface totale (bâti + terrain)
    df["surface_totale"] = df["Surface reelle bati"] + df["Surface terrain"]

    # Ratio bâti/terrain
    df["ratio_bati_terrain"] = df["Surface reelle bati"] / df["Surface terrain"].replace(0,1)

    # Surface moyenne par lot
    df["surface_moy_lot"] = df["surface_lots_total"] / df["nb_lots"].replace(0,1)

    # Features de standing (lots multiples ou très grandes surfaces)
    df["luxury_flag"] = ((df["Surface reelle bati"] > 200) | (df["Surface terrain"] > 500) | (df["nb_lots"] > 2)).astype(int)

    # ------------------------
    # 8️⃣ Colonnes utiles pour ML
    # ------------------------
    keep_cols = [
        "Valeur_fonciere_log",  
        "Surface reelle bati",
        "Nombre pieces principales",
        "Surface terrain",
        "annee_mutation",
        "mois_mutation",
        "surface_log",
        "terrain_log",
        "surface_par_piece",
        "prix_m2_dep",
        "surface_x_prix_dep",
        "terrain_ratio",
        # Features luxe
        "surface_lots_total",
        "nb_lots",
        "surface_totale",
        "ratio_bati_terrain",
        "surface_moy_lot",
        "luxury_flag"
    ] + [c for c in df.columns if c.startswith("Type local_") or c.startswith("Code departement_")]

    df = df[keep_cols].copy()

    return df
