import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from typing import Tuple, Dict, Any

# Fixed seed for reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 5

def split_development_test(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into development and final holdout test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    return X_dev, X_test, y_dev, y_test

def get_cv_strategy() -> RepeatedStratifiedKFold:
    """
    Returns the repeated stratified K-Fold cross-validation strategy.
    """
    return RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, 
        n_repeats=CV_REPEATS, 
        random_state=RANDOM_SEED
    )

def get_split_stats(y_dev: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Returns statistics about the split.
    """
    return {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "dev_samples": len(y_dev),
        "test_samples": len(y_test),
        "dev_class_0": (y_dev == 0).sum(),
        "dev_class_1": (y_dev == 1).sum(),
        "test_class_0": (y_test == 0).sum(),
        "test_class_1": (y_test == 1).sum()
    }
