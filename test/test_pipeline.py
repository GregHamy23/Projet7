import pytest
import numpy as np
import pandas as pd
from collections import Counter
import sys
sys.path.append("..")
from utils import missing_values_table
from sklearn.preprocessing import LabelEncoder
from utils import encode_categorical_columns, business_score, preprocess_data, impute_missing_values
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

# test missing values
def test_missing_values_table():
    # Création d'un DataFrame de test avec des valeurs manquantes
    data = pd.DataFrame({
        "col1": [1, 2, None, None, 5],   # 40% de NaN
        "col2": [None, None, None, 4, 5],  # 60% de NaN (doit être détecté)
        "col3": [1, 2, 3, 4, 5],   # Pas de NaN
    })

    # Exécution de la fonction
    result = missing_values_table(data, threshold=50)

    # Résultat attendu
    expected = pd.DataFrame({
        "% Missing": [60.0, 40.0],
        "Above Threshold": [True, False]
    }, index=["col2", "col1"])  # L'index doit correspondre aux colonnes concernées

    # Comparaison
    pd.testing.assert_frame_equal(result, expected)
    
def test_impute_missing_values():
    # Créer un DataFrame d'exemple
    data = {
        'feature1': [1.0, 2.0, np.nan],
        'feature2': ['A', 'B', 'A'],
        'feature3': [np.nan, 3.0, 4.0]
        }
    features = pd.DataFrame(data)

    # Imputation des valeurs manquantes
    features_imputed = impute_missing_values(features.copy())

    # Vérifier que les valeurs manquantes ont été imputées uniquement pour les colonnes numériques
    assert not features_imputed[['feature1', 'feature3']].isnull().any().any(), "Il y a encore des valeurs manquantes dans les colonnes numériques."
    assert features_imputed['feature2'].dtype == object, "La colonne catégorielle a été modifiée."

def test_encode_categorical_columns():
    # Créer un DataFrame d'exemple
    data = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': ['A', 'B', 'A'],
        'feature3': [4.0, 5.0, 6.0]
    }
    features = pd.DataFrame(data)

    # Encodage one-hot
    features_ohe = encode_categorical_columns(features.copy(), encoding='ohe')
    assert 'feature2_A' in features_ohe.columns, "L'encodage one-hot n'a pas été appliqué correctement."
    assert 'feature2_B' in features_ohe.columns, "L'encodage one-hot n'a pas été appliqué correctement."

    # Encodage label
    features_le = encode_categorical_columns(features.copy(), encoding='le')
    assert features_le['feature2'].dtype == np.int64, "L'encodage label n'a pas été appliqué correctement."

def test_preprocess_data():
    # Créer un DataFrame d'exemple
    data = {
        'SK_ID_CURR': [1001, 1002, 1003],
        'feature1': [1.0, 2.0, np.nan],
        'feature2': ['A', 'B', 'A'],
        'feature3': [np.nan, 3.0, 4.0]
    }
    features = pd.DataFrame(data)

    # Test pour l'encodage one-hot
    processed_features_ohe, ids_ohe = preprocess_data(features.copy(), encoding='ohe')
    assert np.array_equal(ids_ohe, data['SK_ID_CURR']), "Les identifiants ne correspondent pas pour l'encodage one-hot."
    assert 'feature2_A' in processed_features_ohe.columns, "L'encodage one-hot n'a pas été appliqué correctement."
    assert 'feature2_B' in processed_features_ohe.columns, "L'encodage one-hot n'a pas été appliqué correctement."

    # Test pour l'encodage label
    processed_features_le, ids_le = preprocess_data(features.copy(), encoding='le')
    assert np.array_equal(ids_le, data['SK_ID_CURR']), "Les identifiants ne correspondent pas pour l'encodage label."
    assert processed_features_le['feature2'].dtype == np.int64, "L'encodage label n'a pas été appliqué correctement."

    # Test pour un encodage invalide
    with pytest.raises(ValueError, match="Encoding must be either 'ohe' or 'le'"):
        preprocess_data(features.copy(), encoding='invalid')

# test business score
def test_business_score():
    # Cas de test 1: Aucune erreur
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    score = business_score(y_true, y_pred)
    assert score == 1.0, f"Expected score 1.0, but got {score}"

    # Cas de test 2: Quelques faux positifs
    y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
    score = business_score(y_true, y_pred, fp_weight=1, fn_weight=5)
    expected_score = 0.9  # Calculé manuellement
    assert pytest.approx(score, abs=0.01) == expected_score, f"Expected score {expected_score}, but got {score}"

    # Cas de test 3: Quelques faux négatifs
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    score = business_score(y_true, y_pred, fp_weight=1, fn_weight=5)
    expected_score = 0.75  # Calculé manuellement
    assert pytest.approx(score, abs=0.01) == expected_score, f"Expected score {expected_score}, but got {score}"

    # Cas de test 4: Toutes les prédictions sont incorrectes
    y_true = [1, 1, 1, 1]
    y_pred = [0, 0, 0, 0]
    score = business_score(y_true, y_pred, fp_weight=1, fn_weight=5)
    expected_score = 0.0  # Calculé manuellement
    assert pytest.approx(score, abs=0.01) == expected_score, f"Expected score {expected_score}, but got {score}"

# Pour exécuter les tests, utilisez la commande suivante dans le terminal :
# pytest test_pipeline.py

