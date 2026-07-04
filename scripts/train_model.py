import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, clean_data
from src.data.split import split_development_test, get_cv_strategy, RANDOM_SEED
from src.data.preprocess import get_preprocessing_pipeline
from src.features.feature_schema import FEATURES_ORDER

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def run_training():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    model_dir = artifacts_dir / "model"
    plots_dir = artifacts_dir / "plots"
    
    for d in [metrics_dir, model_dir, plots_dir / "calibration", plots_dir / "threshold", plots_dir / "final_evaluation"]:
        d.mkdir(parents=True, exist_ok=True)
        
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    df = clean_data(load_raw_data(raw_data_path))
    X_dev, X_test, y_dev, y_test = split_development_test(df)
    
    from sklearn.model_selection import StratifiedKFold
    cv = get_cv_strategy() # Repeated for training if needed, but not used directly here
    cv_predict = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preprocessor = get_preprocessing_pipeline()

    # Model Benchmarking and Tuning
    models = {
        'LogisticRegression': (
            LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000),
            {'classifier__C': [0.01, 0.1, 1, 10]}
        ),
        'RandomForest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
            {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 5, None], 'classifier__min_samples_split': [2, 5]}
        ),
        'HistGradientBoosting': (
            HistGradientBoostingClassifier(random_state=RANDOM_SEED),
            {'classifier__learning_rate': [0.01, 0.1], 'classifier__max_iter': [50, 100], 'classifier__max_depth': [3, 5]}
        )
    }

    best_models = {}
    comparison_results = []
    tuning_results = []

    for name, (model, params) in models.items():
        pipeline = SklearnPipeline([('preprocessor', preprocessor), ('classifier', model)])
        search = RandomizedSearchCV(pipeline, params, n_iter=5, cv=5, scoring='recall', random_state=RANDOM_SEED, n_jobs=-1)
        search.fit(X_dev, y_dev)
        
        best_pipeline = search.best_estimator_
        best_models[name] = best_pipeline
        
        y_pred = cross_val_predict(best_pipeline, X_dev, y_dev, cv=cv_predict, method='predict', n_jobs=-1)
        try:
            y_proba = cross_val_predict(best_pipeline, X_dev, y_dev, cv=cv_predict, method='predict_proba', n_jobs=-1)[:, 1]
        except AttributeError:
            y_proba = cross_val_predict(best_pipeline, X_dev, y_dev, cv=cv_predict, method='decision_function', n_jobs=-1)

        acc = accuracy_score(y_dev, y_pred)
        bal_acc = balanced_accuracy_score(y_dev, y_pred)
        rec = recall_score(y_dev, y_pred)
        spec = specificity_score(y_dev, y_pred)
        prec = precision_score(y_dev, y_pred)
        f1 = f1_score(y_dev, y_pred)
        roc = roc_auc_score(y_dev, y_proba)
        pr = average_precision_score(y_dev, y_proba)
        
        comparison_results.append({
            'model': name, 'accuracy': acc, 'balanced_accuracy': bal_acc, 'recall': rec,
            'specificity': spec, 'precision': prec, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr
        })
        tuning_results.append({'model': name, 'best_params': str(search.best_params_)})

    pd.DataFrame(comparison_results).to_csv(metrics_dir / "model_comparison.csv", index=False)
    pd.DataFrame(tuning_results).to_csv(metrics_dir / "tuning_results.csv", index=False)
    
    top_candidates = ['LogisticRegression', 'RandomForest']

    # Probability Calibration
    calibration_results = []
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    calibrated_models = {}
    for name in top_candidates:
        pipeline = best_models[name]
        
        y_proba_uncal = cross_val_predict(pipeline, X_dev, y_dev, cv=cv_predict, method='predict_proba', n_jobs=-1)[:, 1]
        brier_uncal = brier_score_loss(y_dev, y_proba_uncal)
        
        calibrated_pipeline = CalibratedClassifierCV(pipeline, method='sigmoid', cv=5)
        y_proba_cal = cross_val_predict(calibrated_pipeline, X_dev, y_dev, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
        brier_cal = brier_score_loss(y_dev, y_proba_cal)
        
        calibrated_pipeline.fit(X_dev, y_dev)
        calibrated_models[name] = calibrated_pipeline
        
        roc_cal = roc_auc_score(y_dev, y_proba_cal)
        pr_cal = average_precision_score(y_dev, y_proba_cal)
        
        calibration_results.append({
            'model': name, 'brier_uncalibrated': brier_uncal, 'brier_calibrated': brier_cal,
            'roc_auc_calibrated': roc_cal, 'pr_auc_calibrated': pr_cal
        })
        
        prob_true, prob_pred = calibration_curve(y_dev, y_proba_cal, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{name} (Sigmoid)")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves (Development Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "calibration" / "calibration_curve.png")
    plt.close()

    pd.DataFrame(calibration_results).to_csv(metrics_dir / "calibration_comparison.csv", index=False)
    
    # AUDIT FIX: Calibration worsened Brier score (0.119 uncal vs 0.123 cal). Reverting to uncalibrated.
    final_model_name = 'LogisticRegression'
    final_pipeline = best_models[final_model_name]
    
    # Decision Threshold Optimization
    y_proba_dev = cross_val_predict(final_pipeline, X_dev, y_dev, cv=cv_predict, method='predict_proba', n_jobs=-1)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 81)
    threshold_results = []
    for t in thresholds:
        y_pred_t = (y_proba_dev >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_dev, y_pred_t, labels=[0,1]).ravel()
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        spec = tn / (tn + fp) if (tn+fp) > 0 else 0
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec+rec) > 0 else 0
        threshold_results.append({
            'threshold': t, 'recall': rec, 'specificity': spec, 'precision': prec, 'f1': f1,
            'false_negatives': fn, 'false_positives': fp,
            'accuracy': (tp+tn)/(tp+tn+fp+fn), 'balanced_accuracy': (rec+spec)/2
        })
    
    t_df = pd.DataFrame(threshold_results)
    t_df.to_csv(metrics_dir / "threshold_analysis.csv", index=False)
    
    candidates = t_df[t_df['recall'] >= 0.85]
    if len(candidates) > 0:
        best_t_row = candidates.sort_values(by='f1', ascending=False).iloc[0]
    else:
        best_t_row = t_df.sort_values(by='recall', ascending=False).iloc[0]
        
    selected_threshold = float(best_t_row['threshold'])
    
    plt.figure(figsize=(6,4))
    plt.plot(t_df['threshold'], t_df['recall'], label='Recall')
    plt.plot(t_df['threshold'], t_df['specificity'], label='Specificity')
    plt.axvline(selected_threshold, color='k', linestyle='--', label=f'Selected ({selected_threshold:.2f})')
    plt.legend()
    plt.title("Recall and Specificity vs Threshold")
    plt.tight_layout()
    plt.savefig(plots_dir / "threshold" / "recall_specificity_curve.png")
    plt.close()
    
    threshold_artifact = {
        'selected_threshold': selected_threshold,
        'selection_policy': 'recall >= 0.85 then max F1',
        'development_recall': float(best_t_row['recall']),
        'development_specificity': float(best_t_row['specificity']),
        'development_precision': float(best_t_row['precision']),
        'development_f1': float(best_t_row['f1']),
        'timestamp': str(datetime.datetime.now())
    }
    with open(model_dir / "decision_threshold.json", "w") as f:
        json.dump(threshold_artifact, f, indent=4)
        
    # Final Held-Out Test Evaluation
    final_pipeline.fit(X_dev, y_dev)
    joblib.dump(final_pipeline, model_dir / "final_pipeline.joblib")
    
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= selected_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0,1]).ravel()
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'specificity': float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_test_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'pr_auc': float(average_precision_score(y_test, y_test_proba)),
        'brier_score': float(brier_score_loss(y_test, y_test_proba)),
        'false_negatives': int(fn),
        'false_positives': int(fp),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    model_metadata = {
        'model_name': 'LogisticRegression_Calibrated',
        'model_version': '1.0.0',
        'training_date': str(datetime.datetime.now()),
        'random_seed': RANDOM_SEED,
        'feature_count': len(FEATURES_ORDER),
        'feature_order': FEATURES_ORDER,
        'preprocessing_summary': 'Imputation, Scaling, OneHotEncoding',
        'model_hyperparameters': str(best_models['LogisticRegression'].steps[1][1].get_params()),
        'imbalance_handling': 'class_weight=balanced natively in LR',
        'calibration_status': 'Uncalibrated (calibration worsened Brier score)',
        'selected_threshold': selected_threshold,
        'dataset_identifier': 'UCI Heart Disease Cleveland'
    }
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)
        
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(plots_dir / "final_evaluation" / "confusion_matrix.png")
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {test_metrics['roc_auc']:.2f}")
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.savefig(plots_dir / "final_evaluation" / "roc_curve.png")
    plt.close()
    
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec, label=f"PR-AUC = {test_metrics['pr_auc']:.2f}")
    plt.title('PR Curve (Test Set)')
    plt.legend()
    plt.savefig(plots_dir / "final_evaluation" / "precision_recall_curve.png")
    plt.close()

if __name__ == "__main__":
    run_training()
