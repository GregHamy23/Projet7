import pytest
import pandas as pd
from collections import Counter
import sys
sys.path.append("..")
from utils import missing_values_table

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

# # Définir une fixture pytest pour charger app_train
# @pytest.fixture
# def X_test_bis():
#     return pd.read_csv('/Users/gregoirehamy/Desktop/OpenClassrooms/Projet7/X_test_bis.csv')

# def test_skewness(X_test_bis, threshold=2):
#     skewness = X_test_bis.skew().abs()
#     assert all(skewness < threshold), "Certaines variables sont trop asymétriques !"

# def test_missing_values(X_test_bis):
#     assert X_test_bis.isnull().sum().sum() < 0.1 * X_test_bis.size, "Trop de valeurs manquantes !"

# def test_no_string_columns(app_train):
#     assert all(X_test_bis.select_dtypes(include=['object']).columns == []), "Des colonnes catégoriques n'ont pas été encodées !"

# @pytest.fixture
# def y_data():
#     # Remplace ces listes par tes vraies données y_train et y_resampled
#     y_train = [0, 1, 1, 0, 0, 1, 1, 0]
#     y_resampled = [0, 1, 1, 0, 1, 1, 0, 1]
#     return y_train, y_resampled

# def test_balanced_classes(y_data):
#     y_train, y_resampled = y_data
#     orig_counts = Counter(y_train)
#     new_counts = Counter(y_resampled)
#     assert min(new_counts.values()) / max(new_counts.values()) > 0.8, "Classes mal équilibrées après SMOTE !"

