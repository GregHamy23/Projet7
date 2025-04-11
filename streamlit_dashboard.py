import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import mlflow.sklearn
import boto3


# Charger le modèle depuis le fichier Pickle
with open('GBmodel.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Charger les données
data = pd.read_csv('results.csv')

# Initialiser l'explainer SHAP pour le modèle
explainer = shap.Explainer(loaded_model, data.drop(columns=['SK_ID_CURR', 'PREDICTION', 'PREDICTION_PROBA']))

# Interface utilisateur
st.title("Tableau de Bord Credit Scoring")

# Sélection du client
client_id = st.selectbox("Sélectionnez un client", data['SK_ID_CURR'])

# Afficher les informations du client sélectionné
client_info = data[data['SK_ID_CURR'] == client_id]
st.write("### Informations du Client")
st.dataframe(client_info)

# Afficher la probabilité et l'éligibilité
st.subheader("Probabilité et Éligibilité")

# Extraire la probabilité
probability = client_info['PREDICTION_PROBA'].values[0]

# Inverser la probabilité pour la jauge
inverted_probability = 1 - probability

# Afficher la probabilité avec une jauge colorée
st.metric(label="Probabilité de non remboursement", value=f"{probability:.2%}")

# Jauge personnalisée avec HTML/CSS
if inverted_probability >= 0.75:
    color = 'green'
elif inverted_probability >= 0.5:
    color = 'orange'
else:
    color = 'red'

progress_html = f"""
<div style="width:100%; background-color:#e0e0e0; border-radius:5px; overflow:hidden;">
    <div style="width:{int(inverted_probability * 100)}%; background-color:{color}; color:white; text-align:center; padding:10px 0; border-radius:5px;">
        {int(probability * 100)}%
    </div>
</div>
"""
st.markdown(progress_html, unsafe_allow_html=True)

# Déterminer l'éligibilité et afficher avec couleur
if probability >= 0.5:
    st.markdown("<p style='color: red; font-weight: bold;'>Non éligible à un prêt</p>", unsafe_allow_html=True)
    st.write("La probabilité de non remboursement de prêt est supérieure à 50%, ce qui indique que le client n'est pas éligible à un prêt.")
else:
    st.markdown("<p style='color: green; font-weight: bold;'>Éligible à un prêt</p>", unsafe_allow_html=True)
    st.write("La probabilité de non remboursement de prêt est inférieure à 50%, ce qui indique que le client est éligible à un prêt.")

# Visualisation de l'importance des features
st.subheader("Importance des Features")

# Calculer les valeurs SHAP pour le client sélectionné
shap_values = explainer(client_info.drop(columns=['SK_ID_CURR', 'PREDICTION', 'PREDICTION_PROBA']))

# Afficher l'importance locale des features
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Afficher l'importance globale des features
fig, ax = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

# Ajouter un graphique pour comparer la probabilité du client sélectionné avec les autres
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['PREDICTION_PROBA'], kde=True, bins=30, color='blue', ax=ax)

# Ajouter une ligne verticale pour la probabilité du client sélectionné
client_probability = client_info['PREDICTION_PROBA'].values[0]
ax.axvline(x=client_probability, color='red', linestyle='--', label=f'Probabilité du Client {client_id}')

ax.set_title("Comparaison de la Probabilité du Client avec les Autres Clients")
ax.set_xlabel("Probabilité")
ax.set_ylabel("Fréquence")
ax.legend()
st.pyplot(fig)

# Afficher les statistiques descriptives
st.subheader("Statistiques Descriptives du Client")
st.write(client_info.describe())

# Comparaison avec d'autres clients
st.subheader("Comparaison avec d'autres clients")

# Sélection des caractéristiques pour la comparaison
features_to_compare = st.multiselect("Sélectionnez les caractéristiques à comparer", data.columns)

# Filtrer les données pour les caractéristiques sélectionnées
if features_to_compare:
    comparison_data = data[features_to_compare]

    # Créer un graphique de comparaison avec boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=comparison_data, ax=ax)

    # Ajouter des annotations pour le client sélectionné
    for feature in features_to_compare:
        client_value = client_info[feature].values[0]
        ax.axvline(x=client_value, color='red', linestyle='--', label=f'Client {client_id}')

    ax.set_title(f"Comparaison des caractéristiques pour le client {client_id}")
    ax.legend()
    st.pyplot(fig)

    # Ajouter un graphique de distribution avec histplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=comparison_data, kde=True, ax=ax)

    # Ajouter des annotations pour le client sélectionné
    for feature in features_to_compare:
        client_value = client_info[feature].values[0]
        ax.axvline(x=client_value, color='red', linestyle='--', label=f'Client {client_id}')

    ax.set_title(f"Distribution des caractéristiques pour le client {client_id}")
    ax.legend()
    st.pyplot(fig)

# Graphique d’analyse bi-variée
st.subheader("Analyse Bi-variée")
feature_1 = st.selectbox("Sélectionnez la première caractéristique", data.columns)
feature_2 = st.selectbox("Sélectionnez la deuxième caractéristique", data.columns)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data[feature_1], y=data[feature_2], ax=ax)
ax.set_title(f"Analyse bi-variée entre {feature_1} et {feature_2}")
ax.set_xlabel(feature_1)
ax.set_ylabel(feature_2)
st.pyplot(fig)

# Fonctionnalités optionnelles
st.sidebar.title("Nouvelle simulation")

# Ajouter des champs de saisie pour modifier les informations du client
st.sidebar.subheader("Modifier les Informations du Client")
new_income = st.sidebar.number_input("Nouveau Revenu Total", value=client_info['AMT_INCOME_TOTAL'].values[0])
new_credit = st.sidebar.number_input("Nouveau Crédit", value=client_info['AMT_CREDIT'].values[0])
new_annuity = st.sidebar.number_input("Nouvelle annuité", value=client_info['AMT_ANNUITY'].values[0])

# Recalculer le score et la probabilité
if st.sidebar.button("Recalculer la Probabilité"):
    # Mettre à jour les données du client avec les nouvelles valeurs
    updated_client_info = client_info.copy()
    updated_client_info['AMT_INCOME_TOTAL'] = new_income
    updated_client_info['AMT_CREDIT'] = new_credit
    updated_client_info['AMT_ANNUITY'] = new_annuity

    # Préparer les données pour la prédiction
    input_data = updated_client_info.drop(columns=['SK_ID_CURR', 'PREDICTION', 'PREDICTION_PROBA']).values.reshape(1, -1)

    # Recalculer la probabilité avec le modèle
    updated_probability = loaded_model.predict_proba(input_data)[0][1]

    # Afficher la nouvelle probabilité dans la sidebar
    st.sidebar.metric(label="Nouvelle Probabilité de non remboursement", value=f"{updated_probability:.2%}")

    # Déterminer l'éligibilité et afficher avec couleur dans la sidebar
    if updated_probability >= 0.5:
        st.sidebar.markdown("<p style='color: red; font-weight: bold;'>Non éligible à un prêt</p>", unsafe_allow_html=True)
        st.sidebar.write("La probabilité de non remboursement de prêt est supérieure à 50%, ce qui indique que le client n'est pas éligible à un prêt.")
    else:
        st.sidebar.markdown("<p style='color: green; font-weight: bold;'>Éligible à un prêt</p>", unsafe_allow_html=True)
        st.sidebar.write("La probabilité de non remboursement de prêt est inférieure à 50%, ce qui indique que le client est éligible à un prêt.")
