import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys
from pathlib import Path

# Add project root to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.feature_schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES, DISCRETE_FEATURES

def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for leakage-safe preprocessing.
    Numeric features are imputed with median and scaled.
    Categorical features are imputed with most_frequent and one-hot encoded.
    Discrete features are imputed with most_frequent and scaled.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    discrete_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('disc', discrete_transformer, DISCRETE_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor
