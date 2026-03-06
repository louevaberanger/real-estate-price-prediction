import pandas as pd

def clean_data(df):
    # convertir Valeur fonciere en float
    df["Valeur fonciere"] = df["Valeur fonciere"].str.replace(",", ".")
    df["Valeur fonciere"] = pd.to_numeric(df["Valeur fonciere"], errors="coerce")

    # supprimer lignes sans valeurs essentielles
    df = df[df["Valeur fonciere"].notna()]
    df = df[df["Surface reelle bati"].notna()]
    df = df[df["Surface reelle bati"] > 0]
    df = df[df["Valeur fonciere"] > 10000]

    # filtrer Surface et prix au m²
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    df = df[df["prix_m2"] < 20000]
    df = df[df["prix_m2"] > 500]
    df = df[df["Surface reelle bati"] < 1000]

    # remplir les NaN
    df["Nombre pieces principales"].fillna(df["Nombre pieces principales"].median(), inplace=True)
    df["Surface terrain"].fillna(0, inplace=True)
    df["Code postal"] = df["Code postal"].fillna(0)

    # Date mutation → année / mois
    df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")
    df["annee_mutation"] = df["Date mutation"].dt.year
    df["mois_mutation"] = df["Date mutation"].dt.month

    # extraire département depuis Code postal
    df["Code departement"] = df["Code postal"].astype(str).str[:2]

    # encodage des colonnes catégorielles
    df = pd.get_dummies(df, columns=["Type local", "Code departement"], drop_first=True)

    # garder seulement les colonnes utiles pour ML
    keep_cols = ["Valeur fonciere", "Surface reelle bati", "Nombre pieces principales",
                 "Surface terrain", "annee_mutation", "mois_mutation"] + \
                [c for c in df.columns if c.startswith("Type local_") or c.startswith("Code departement_")]
    df = df[keep_cols].copy()

    return df