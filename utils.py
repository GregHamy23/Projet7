from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def impute_missing_values(features):
    # Impute missing values only for numerical columns
    numerical_features = features.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="mean")
    numerical_features_imputed = imputer.fit_transform(numerical_features)

    # Replace the numerical columns in the original DataFrame
    features[numerical_features.columns] = numerical_features_imputed
    return features

def encode_categorical_columns(features, encoding='ohe'):
    # Encode categorical columns
    if encoding == 'ohe':
        features = pd.get_dummies(features, columns=features.select_dtypes(include=[object]).columns)
    elif encoding == 'le':
        label_encoder = LabelEncoder()
        for col in features.select_dtypes(include=[object]).columns:
            features[col] = label_encoder.fit_transform(features[col].astype(str))
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    return features

def preprocess_data(features, encoding='ohe'):
    # Extract the ids
    ids = features['SK_ID_CURR']

    # Remove the ids
    features = features.drop(columns=['SK_ID_CURR'])

    # Impute missing values
    features = impute_missing_values(features)

    # Encode categorical columns
    features = encode_categorical_columns(features, encoding)

    # Add the ids back to the dataframe
    features['SK_ID_CURR'] = ids

    print('Training Data Shape: ', features.shape)
    return features, ids


# test business score
def business_score(y_true, y_pred, fp_weight=1, fn_weight=5):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    penalty = fp_weight * fp + fn_weight * fn

    # Calculer la pénalité maximale possible
    max_penalty = len(y_true) * max(fp_weight, fn_weight)

    # Normaliser la pénalité entre 0 et 1
    normalized_penalty = penalty / max_penalty if max_penalty > 0 else 0

    # Inverser la pénalité pour obtenir un score entre 0 et 1
    normalized_score = 1 - normalized_penalty

    return normalized_score

# test missing values
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