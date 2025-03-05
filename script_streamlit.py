#streamlit run script_streamlit.py

import streamlit as st
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer

# Spécifiez le nom du modèle et la version que vous souhaitez récupérer
mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_name = "RandomForestClassifier"
model_version = 2

# Récupérer le modèle depuis MLflow
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.sklearn.load_model(model_uri)
print(f"Modèle chargé : {model_name} version {model_version}")

# Titre de l'application
st.title('Application de Prédiction remboursement de prêt')

# Description de l'application
st.write("""
Cette application permet de faire des prédictions en mettant en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé.
Chargez un fichier CSV contenant les données de test et cliquez sur le bouton 'Prédire' pour obtenir les résultats.
""")

# Widget pour charger un fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Vérifier si un fichier a été chargé
if uploaded_file is not None:
    # Lire le fichier CSV
    test_df = pd.read_csv(uploaded_file)

    # Afficher les premières lignes du DataFrame
    st.write("Aperçu des données chargées :")
    st.write(test_df.head())

    # Appliquer l'encoding
    for col in test_df.select_dtypes(include=['object']).columns:
        if len(test_df[col].unique()) <= 2:
            try:
                # Charger l'encoder pour cette colonne
                with open(f'label_encoder_{col}.pkl', 'rb') as le_file:
                    le = pickle.load(le_file)
                # Transformer les données
                test_df[col] = le.transform(test_df[col])
            except FileNotFoundError:
                st.error(f"Aucun encoder trouvé pour la colonne: {col}")

    # One-hot encoding
    test_df_dummy = pd.get_dummies(test_df)

    # Charger l'imputer
    imputer = SimpleImputer(strategy="mean")  
    test_df_imputed = imputer.fit_transform(test_df_dummy)  
    test_df_imputed

    # Imputer
    test_df_imputed = imputer.transform(test_df_dummy)

    # Bouton pour exécuter les prédictions
    if st.button('Prédire'):
        # Effectuer les prédictions
        y_pred_proba = loaded_model.predict_proba(test_df_imputed)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Convertir les données imputées en DataFrame
        test_df_imputed = pd.DataFrame(test_df_imputed, columns=test_df_dummy.columns)

        # Ajouter les prédictions au DataFrame de test
        test_df_imputed['PREDICTION'] = y_pred
        test_df_imputed['PREDICTION_PROBA'] = y_pred_proba

        # Sélectionner les colonnes à afficher
        result_df = test_df_imputed[['SK_ID_CURR', 'PREDICTION', 'PREDICTION_PROBA']]

        # Convertir la colonne SK_ID_CURR en entier
        result_df['SK_ID_CURR'] = result_df['SK_ID_CURR'].astype(int)

        # Afficher les résultats
        st.write("Résultats des prédictions :")
        st.write(result_df)