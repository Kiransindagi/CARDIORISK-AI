import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_schema import FEATURES_ORDER, TARGET, CATEGORY_MAPPING
from src.data.load_data import load_raw_data, clean_data
from src.data.split import split_development_test, get_cv_strategy, RANDOM_SEED
from src.data.preprocess import get_preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE

@pytest.fixture(scope="module")
def df():
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    return clean_data(load_raw_data(raw_data_path))

def test_feature_schema():
    assert len(FEATURES_ORDER) == 13
    assert TARGET not in FEATURES_ORDER
    assert "age" in FEATURES_ORDER
    assert "sex" in CATEGORY_MAPPING
    assert list(CATEGORY_MAPPING["sex"].values()) == ["Female", "Male"]

def test_data_split(df):
    X_dev, X_test, y_dev, y_test = split_development_test(df)
    
    # Reproducibility
    X_dev2, X_test2, y_dev2, y_test2 = split_development_test(df)
    assert X_dev.equals(X_dev2)
    
    # No overlap
    assert len(set(X_dev.index).intersection(set(X_test.index))) == 0
    
    # Stratification (proportions roughly equal)
    dev_prop = y_dev.mean()
    test_prop = y_test.mean()
    assert abs(dev_prop - test_prop) < 0.05
    
    # Target not in X
    assert TARGET not in X_dev.columns

def test_preprocessing(df):
    X_dev, _, y_dev, _ = split_development_test(df)
    preprocessor = get_preprocessing_pipeline()
    
    # Fit and transform
    X_trans = preprocessor.fit_transform(X_dev)
    
    # Output is numpy array, no missing values
    assert np.isnan(X_trans).sum() == 0
    
    # Unknown category handling
    X_test_fake = X_dev.iloc[[0]].copy()
    X_test_fake['chest_pain_type'] = 999.0 # Unknown category
    X_trans_fake = preprocessor.transform(X_test_fake)
    assert np.isnan(X_trans_fake).sum() == 0

def test_baseline_pipeline(df):
    X_dev, X_test, y_dev, y_test = split_development_test(df)
    preprocessor = get_preprocessing_pipeline()
    model = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = SklearnPipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_dev, y_dev)
    probs = pipeline.predict_proba(X_test)
    
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

def test_imbalance_pipeline(df):
    X_dev, X_test, y_dev, y_test = split_development_test(df)
    preprocessor = get_preprocessing_pipeline()
    smote = SMOTE(random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    pipeline = ImblearnPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', model)
    ])
    
    pipeline.fit(X_dev, y_dev)
    
    # Pipeline executes successfully on test set without resampling
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    
    # Ensure SMOTE is only in ImblearnPipeline
    assert 'smote' in pipeline.named_steps
