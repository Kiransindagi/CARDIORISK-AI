import pandas as pd
import numpy as np
from pathlib import Path

# Mapping of original columns to readable names
COLUMN_MAPPING = {
    "age": "age",
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "num_major_vessels",
    "thal": "thalassemia",
    "num": "target"
}

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw Cleveland dataset, set column names, and replace '?' with NaN.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # The dataset does not have a header
    columns = list(COLUMN_MAPPING.keys())
    
    df = pd.read_csv(filepath, names=columns, na_values=["?"])
    
    # Rename columns to human-readable names
    df = df.rename(columns=COLUMN_MAPPING)
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dataset-specific cleaning:
    - target transformation (0 -> 0, 1,2,3,4 -> 1)
    - converting certain numeric encodings to strings (optional here, but we will keep as is or map later in features)
    """
    df_cleaned = df.copy()
    
    # Target transformation
    # 0 = absence of heart disease
    # 1, 2, 3, 4 = presence of heart disease
    df_cleaned['target'] = df_cleaned['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df_cleaned
