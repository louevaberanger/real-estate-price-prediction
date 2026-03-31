import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
model = joblib.load("models/price_model_lgbm_luxury.pkl")  # c'est un Booster

# Récupérer les noms des features et leur importance
features = model.feature_name() if hasattr(model, "feature_name") else model.feature_name()  # attribut
importances = model.feature_importance()

# Créer un DataFrame trié
df_importances = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Afficher les 20 features les plus importantes
print(df_importances.head(20))

# Plot des 20 plus importantes
plt.figure(figsize=(10,6))
plt.barh(
    df_importances["feature"].head(20)[::-1],
    df_importances["importance"].head(20)[::-1]
)
plt.xlabel("Importance")
plt.title("Top 20 features les plus influentes sur le prix")
plt.show()
