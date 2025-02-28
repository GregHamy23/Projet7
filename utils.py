import pandas as pd
def missing_values_table(df, threshold=50):
    """Retourne les colonnes ayant des valeurs manquantes avec leur pourcentage."""
    mis_val_percent = df.isnull().mean() * 100
    missing_df = mis_val_percent[mis_val_percent > 0].to_frame(name="% Missing").sort_values(by="% Missing", ascending=False)

    # Affichage du résumé
    print(f"Le dataframe contient {df.shape[1]} colonnes.")
    print(f"{missing_df.shape[0]} colonnes ont des valeurs manquantes.")

    # Indicateur de colonnes problématiques
    missing_df["Above Threshold"] = missing_df["% Missing"] > threshold

    return missing_df