#streamlit run script_streamlit.py

import streamlit as st
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
import mlflow
import boto3

# Titre de l'application
st.title('Application de Prédiction remboursement de prêt')

# Description de l'application
st.write("""
Cette application permet de faire des prédictions en mettant en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé.
Chargez un fichier CSV contenant les données de test et cliquez sur le bouton 'Prédire' pour obtenir les résultats.
""")

# Widget pour charger un fichier CSV
# @st.cache_data
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Vérifier si un fichier a été chargé
if uploaded_file is not None:
    # Lire le fichier CSV
    test_df = pd.read_csv(uploaded_file)

    # Afficher les premières lignes du DataFrame
    st.write("Aperçu des données chargées :")
    st.write(test_df.head())

    # Configurer MLflow pour utiliser le serveur distant avec les identifiants spécifiés
    mlflow.set_tracking_uri("http://ec2-51-20-85-239.eu-north-1.compute.amazonaws.com:5000/")

        # Run ID
    run_id = "2c7d72ff06314a02b8d516519a1fcb6f"

    # Charger le modèle qui contient tous les encoders
    encoders_model_uri = f"runs:/{run_id}/label_encoders"
    encoders = mlflow.sklearn.load_model(encoders_model_uri)

    # Appliquer l'encoding
    le_count = 0

    for col, encoder in encoders.items():
        if col in test_df.columns:
            try:
                # Transformer les données
                test_df[col] = encoder.transform(test_df[col])
                # Incrémenter le compteur
                le_count += 1
            except Exception as e:
                # Afficher l'erreur dans le notebook
                print(f"Erreur lors du chargement de l'encoder pour la colonne {col}: {e}")

    # One-hot encoding
    test_df = pd.get_dummies(test_df)
    def preprocess_data(features, encoding='ohe'):

        # Extract the ids
        ids = features['SK_ID_CURR']

        # Remove the ids 
        features = features.drop(columns=['SK_ID_CURR'])

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        features = imputer.fit_transform(features)

        # One Hot Encoding
        if encoding == 'ohe':
            features = pd.get_dummies(pd.DataFrame(features))

            # No categorical indices to record
            cat_indices = 'auto'

        # Integer label encoding
        elif encoding == 'le':
            label_encoder = LabelEncoder()
            cat_indices = []

            # Iterate through each column
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape(-1,))
                    cat_indices.append(col)

        # Catch error if label encoding scheme is not valid
        else:
            raise ValueError("Encoding must be either 'ohe' or 'le'")
        
        # Add the ids back to the dataframes
        features['SK_ID_CURR'] = ids

        print('Training Data Shape: ', features.shape)

        return features, ids, cat_indices
    
    features, ids, cat_indices = preprocess_data(test_df)
    
    # Modèle à récupérer
    model_name = "GradientBoosting"  
    model_version = 4

    # Récupérer le modèle depuis MLflow
    model_uri = f"models:/{model_name}/{model_version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Vérifier si les prédictions ont déjà été faites
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False

    # Bouton pour exécuter les prédictions
    if st.button('Prédire'):
        
        # Convertir les noms de colonnes en chaînes de caractères
        features.columns = features.columns.astype(str)
        # Vérifier les types de colonnes après conversion
        print(features.dtypes)
        test_df_final=features.copy()
        
        # Effectuer les prédictions
        y_pred_proba = loaded_model.predict_proba(features)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Ajouter les prédictions au DataFrame de test
        test_df_final['PREDICTION'] = y_pred
        test_df_final['PREDICTION_PROBA'] = y_pred_proba

        # Sélectionner les colonnes à afficher
        result_df = test_df_final[['SK_ID_CURR', 'PREDICTION', 'PREDICTION_PROBA']]

        # Stocker les résultats dans st.session_state
        st.session_state.result_df = result_df
        st.session_state.test_df_imputed = features
        st.session_state.predictions_made = True

        # Afficher les résultats
        st.write("Résultats des prédictions :")
        st.write(result_df)

        # Créer un échantillon pour la performance
        background_sample = st.session_state.features.iloc[:100]

        # SHAP Explainer
        explainer = shap.Explainer(loaded_model)
        shap_values = explainer(background_sample)

        # Initialize the SHAP JavaScript library
        shap.initjs()

        # Feature importance globale
        shap.plots.waterfall(shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=background_sample.iloc[0]), max_display=20)
        plt.title(f"Global SHAP explaination")
        st.pyplot(plt.gcf())

    # Si les prédictions ont été faites, permettre la sélection pour l'explication SHAP
    if st.session_state.predictions_made:
        # Widget pour sélectionner un SK_ID_CURR
        SK_ID_CURR = st.selectbox("Sélectionnez un SK_ID_CURR pour voir l'explication SHAP :", st.session_state.result_df['SK_ID_CURR'])

        # Fonction pour afficher l'explication SHAP
        def show_shap_explanation(SK_ID_CURR):
            
            # Vérifier si le SK_ID_CURR existe dans le DataFrame
            if SK_ID_CURR not in background_sample['SK_ID_CURR'].values:
                st.error(f"Erreur: SK_ID_CURR {SK_ID_CURR} non trouvé dans les données.")
                return

            # Trouver l'index de l'individu sélectionné
            index = background_sample[background_sample['SK_ID_CURR'] == SK_ID_CURR].index[0]

            # SHAP Explainer
            explainer = shap.KernelExplainer(loaded_model.predict_proba, background_sample)
            shap_values = explainer.shap_values(background_sample.iloc[[index]])

            # Utiliser shap.plots pour générer une visualisation statique
            shap.plots.force(explainer.expected_value[1], shap_values[0, :, 1], background_sample.iloc[index], matplotlib=True)
            plt.title(f"SHAP Explanation for SK_ID_CURR: {SK_ID_CURR}")
            st.pyplot(plt.gcf())

        # Afficher l'explication SHAP pour le SK_ID_CURR saisi
        show_shap_explanation(SK_ID_CURR)
