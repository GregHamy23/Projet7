import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import mlflow.sklearn

# Charger le modèle depuis le fichier Pickle
with open('GBmodel.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Charger les encodeurs depuis le fichier Pickle
with open('all_encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

# Charger les données
data = pd.read_csv('results.csv')

# Appliquer l'encoding aux colonnes spécifiées
for col, encoder in encoders.items():
    if col in data.columns:
        data[col] = encoder.transform(data[col])

# Interface utilisateur
st.title("Tableau de Bord Interactif des Clients")

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

# Afficher la probabilité
st.metric(label="Probabilité de non remboursement", value=f"{probability:.2%}")

# Déterminer l'éligibilité et afficher avec couleur
if probability >= 0.5:
    st.markdown("<p style='color: red; font-weight: bold;'>Non éligible à un prêt</p>", unsafe_allow_html=True)
    st.write("La probabilité de non remboursement de prêt est supérieure à 50%, ce qui indique que le client n'est pas éligible à un prêt.")
else:
    st.markdown("<p style='color: green; font-weight: bold;'>Éligible à un prêt</p>", unsafe_allow_html=True)
    st.write("La probabilité de non remboursement de prêt est inférieure à 50%, ce qui indique que le client est éligible à un prêt.")

# Ajouter un graphique pour comparer la probabilité du client sélectionné avec les autres
plt.figure(figsize=(10, 6))
sns.histplot(data['PREDICTION_PROBA'], kde=True, bins=30, color='blue')

# Ajouter une ligne verticale pour la probabilité du client sélectionné
client_probability = client_info['PREDICTION_PROBA'].values[0]
plt.axvline(x=client_probability, color='red', linestyle='--', label=f'Probabilité du Client {client_id}')

plt.title("Comparaison de la Probabilité du Client avec les Autres Clients")
plt.xlabel("Probabilité")
plt.ylabel("Fréquence")
plt.legend()
st.pyplot(plt)

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
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=comparison_data)

    # Ajouter des annotations pour le client sélectionné
    for feature in features_to_compare:
        client_value = client_info[feature].values[0]
        plt.axvline(x=client_value, color='red', linestyle='--', label=f'Client {client_id}')

    plt.title(f"Comparaison des caractéristiques pour le client {client_id}")
    plt.legend()
    st.pyplot(plt)

    # Optionnel : Ajouter un graphique de distribution avec histplot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=comparison_data, kde=True)

    # Ajouter des annotations pour le client sélectionné
    for feature in features_to_compare:
        client_value = client_info[feature].values[0]
        plt.axvline(x=client_value, color='red', linestyle='--', label=f'Client {client_id}')

    plt.title(f"Distribution des caractéristiques pour le client {client_id}")
    plt.legend()
    st.pyplot(plt)

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
