import pandas as pd

def load_dvf_data(file_path):
    """
    Charge les données DVF depuis un fichier CSV et retourne un DataFrame pandas.
    
    Args:
        file_path (str): chemin vers le fichier CSV
    Returns:
        df (pd.DataFrame)
    """
    df = pd.read_csv(file_path, sep='|', low_memory=False)
    return df