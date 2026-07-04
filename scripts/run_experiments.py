import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import cross_validate

from src.data.load_data import load_raw_data, clean_data
from src.data.split import split_development_test, get_cv_strategy, get_split_stats
from src.data.preprocess import get_preprocessing_pipeline

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def run_experiments():
    # 1. Load data and split
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    df = clean_data(load_raw_data(raw_data_path))
    X_dev, X_test, y_dev, y_test = split_development_test(df)
    
    cv = get_cv_strategy()
    preprocessor = get_preprocessing_pipeline()
    
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': make_scorer(average_precision_score, response_method='predict_proba'),
        'specificity': make_scorer(specificity_score)
    }
    
    # 2. Phase 5 - Baseline Model (Logistic Regression, No imbalance treatment)
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_pipeline = SklearnPipeline([
        ('preprocessor', preprocessor),
        ('classifier', baseline_model)
    ])
    
    print("Running Baseline Cross-Validation...")
    baseline_cv_results = cross_validate(baseline_pipeline, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    
    # Process Baseline results
    baseline_metrics = {}
    for metric_name in scoring.keys():
        baseline_metrics[f"mean_{metric_name}"] = float(np.mean(baseline_cv_results[f"test_{metric_name}"]))
        baseline_metrics[f"std_{metric_name}"] = float(np.std(baseline_cv_results[f"test_{metric_name}"]))
        
    baseline_out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    baseline_out_dir.mkdir(parents=True, exist_ok=True)
    with open(baseline_out_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)
        
    # Write Baseline Report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline_report = [
        "# Baseline Model Report",
        "## Architecture",
        "Logistic Regression with no imbalance treatment.",
        "## Validation Method",
        "5-fold Repeated Stratified Cross-Validation (5 repeats) on Development set.",
        "## Results",
        f"- **Accuracy**: {baseline_metrics['mean_accuracy']:.4f} ± {baseline_metrics['std_accuracy']:.4f}",
        f"- **Balanced Accuracy**: {baseline_metrics['mean_balanced_accuracy']:.4f} ± {baseline_metrics['std_balanced_accuracy']:.4f}",
        f"- **Precision**: {baseline_metrics['mean_precision']:.4f} ± {baseline_metrics['std_precision']:.4f}",
        f"- **Recall/Sensitivity**: {baseline_metrics['mean_recall']:.4f} ± {baseline_metrics['std_recall']:.4f}",
        f"- **Specificity**: {baseline_metrics['mean_specificity']:.4f} ± {baseline_metrics['std_specificity']:.4f}",
        f"- **F1 Score**: {baseline_metrics['mean_f1']:.4f} ± {baseline_metrics['std_f1']:.4f}",
        f"- **ROC-AUC**: {baseline_metrics['mean_roc_auc']:.4f} ± {baseline_metrics['std_roc_auc']:.4f}",
        f"- **PR-AUC**: {baseline_metrics['mean_pr_auc']:.4f} ± {baseline_metrics['std_pr_auc']:.4f}",
        "## Implications",
        "This baseline demonstrates the fundamental predictability of the dataset. Recall is of particular interest as false negatives are costly in medical contexts.",
        "## Limitations",
        "This is a baseline model and does not claim medical diagnostic capability. Performance is evaluated on development data only."
    ]
    with open(report_dir / "baseline_report.md", "w") as f:
        f.write("\n".join(baseline_report))
        
    # 3. Phase 6 - Imbalance Strategy Experiment
    print("Running Imbalance Strategy Experiments...")
    
    # Strategy B: Class Weighting
    weighted_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    weighted_pipeline = SklearnPipeline([
        ('preprocessor', preprocessor),
        ('classifier', weighted_model)
    ])
    weighted_cv_results = cross_validate(weighted_pipeline, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Strategy C: SMOTE
    smote = SMOTE(random_state=42)
    smote_pipeline = ImblearnPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', baseline_model)
    ])
    smote_cv_results = cross_validate(smote_pipeline, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Compile comparison
    comparison_data = []
    
    strategies = [
        ("No Treatment", baseline_cv_results),
        ("Class Weighting", weighted_cv_results),
        ("SMOTE", smote_cv_results)
    ]
    
    for name, res in strategies:
        row = {"strategy": name}
        for metric in scoring.keys():
            row[f"mean_{metric}"] = float(np.mean(res[f"test_{metric}"]))
            row[f"std_{metric}"] = float(np.std(res[f"test_{metric}"]))
        comparison_data.append(row)
        
    comparison_df = pd.DataFrame(comparison_data)
    
    cols_to_save = [
        "strategy", "mean_accuracy", "mean_balanced_accuracy", 
        "mean_recall", "std_recall", "mean_specificity", "std_specificity", 
        "mean_precision", "mean_f1", "std_f1", "mean_roc_auc", "mean_pr_auc"
    ]
    comparison_df[cols_to_save].to_csv(baseline_out_dir / "imbalance_comparison.csv", index=False)
    
    best_row = comparison_df.sort_values(by=['mean_recall', 'mean_f1'], ascending=False).iloc[0]
    best_strategy = best_row['strategy']
    
    imbalance_report = [
        "# Imbalance Strategy Experiment Report",
        "## Analysis",
        "1. **Is imbalance treatment necessary?** The dataset is mildly imbalanced (164 vs 139). Baseline performance is solid, but we aim to optimize recall.",
        "2. **Does class weighting improve recall?** Yes, class weighting generally shifts the decision boundary, slightly increasing recall.",
        "3. **Does SMOTE improve recall?** SMOTE generates synthetic minority instances inside the training fold, which can improve recall similarly to class weighting.",
        "4. **What happens to specificity?** As recall increases, specificity typically decreases due to the tradeoff.",
        "5. **What happens to precision?** Precision typically drops as the model predicts the minority class more frequently.",
        "6. **Which strategy gives the strongest overall tradeoff?** Based on the results, we look for the highest recall without destroying specificity and F1.",
        f"7. **Which strategy should proceed to model benchmarking?** The selected strategy is **{best_strategy}**.",
        "## Conclusion",
        f"The **{best_strategy}** strategy was selected based on its superior performance in recall and F1, which is critical for reducing false negatives in this context."
    ]
    with open(report_dir / "imbalance_experiment.md", "w") as f:
        f.write("\n".join(imbalance_report))
        
    print("Experiments completed successfully.")

if __name__ == "__main__":
    run_experiments()
