import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, clean_data
from src.data.split import split_development_test
from src.features.feature_schema import CATEGORY_MAPPING

def get_age_group(age):
    if age < 45:
        return 'under 45'
    elif 45 <= age <= 54:
        return '45-54'
    elif 55 <= age <= 64:
        return '55-64'
    else:
        return '65 and above'

def run_subgroup_analysis():
    print("Phase 13: Subgroup Performance Analysis")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    metrics_dir = artifacts_dir / "metrics"
    reports_dir = PROJECT_ROOT / "reports"
    
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    df = clean_data(load_raw_data(raw_data_path))
    _, X_test, _, y_test = split_development_test(df)
    
    model = joblib.load(artifacts_dir / "model" / "final_pipeline.joblib")
    with open(artifacts_dir / "model" / "decision_threshold.json", "r") as f:
        threshold = json.load(f)['selected_threshold']
        
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    test_results = X_test.copy()
    test_results['y_true'] = y_test.values
    test_results['y_pred'] = y_test_pred
    
    # Subgroups
    test_results['age_group'] = test_results['age'].apply(get_age_group)
    
    # Reverse map sex
    inv_sex = {v: k for k, v in CATEGORY_MAPPING['sex'].items()}
    test_results['sex_label'] = test_results['sex'].map(inv_sex)
    
    subgroups = [
        ('Sex', 'sex_label', ['Female', 'Male']),
        ('Age', 'age_group', ['under 45', '45-54', '55-64', '65 and above'])
    ]
    
    results = []
    
    for group_name, col_name, categories in subgroups:
        for cat in categories:
            subset = test_results[test_results[col_name] == cat]
            if len(subset) == 0:
                continue
                
            y_t = subset['y_true']
            y_p = subset['y_pred']
            
            pos_count = int(y_t.sum())
            neg_count = int(len(y_t) - pos_count)
            
            acc = accuracy_score(y_t, y_p)
            
            # Precision, recall, f1, specificity
            if len(np.unique(y_t)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
                rec = tp / (tp + fn) if (tp+fn)>0 else None
                spec = tn / (tn + fp) if (tn+fp)>0 else None
                prec = tp / (tp + fp) if (tp+fp)>0 else None
                if prec is not None and rec is not None and (prec+rec)>0:
                    f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = None
            else:
                # Edge cases where only one class exists in the subgroup
                rec = 1.0 if (pos_count > 0 and (y_p == 1).all()) else (0.0 if pos_count > 0 else None)
                spec = 1.0 if (neg_count > 0 and (y_p == 0).all()) else (0.0 if neg_count > 0 else None)
                prec = 1.0 if ((y_p == 1).all() and pos_count > 0) else (0.0 if (y_p == 1).any() else None)
                if prec is not None and rec is not None and (prec+rec)>0:
                    f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = None
            
            results.append({
                'category_type': group_name,
                'category_value': cat,
                'sample_count': len(subset),
                'positive_target_count': pos_count,
                'negative_target_count': neg_count,
                'accuracy': acc,
                'recall': rec,
                'specificity': spec,
                'precision': prec,
                'f1': f1
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(metrics_dir / "subgroup_performance.csv", index=False)
    
    report_lines = [
        "# Subgroup Performance Analysis",
        "",
        "**Important Disclaimer**: This analysis is strictly descriptive. The subgroup sizes from the final test set are extremely small, meaning these results are statistically unstable. This analysis cannot establish fairness or an absence of bias. Larger external datasets are required before making any population-level claims.",
        "",
        "## Subgroup Metrics"
    ]
    
    for _, row in df_results.iterrows():
        report_lines.append(f"### {row['category_type']}: {row['category_value']}")
        report_lines.append(f"- **Sample count**: {row['sample_count']} (Positives: {row['positive_target_count']}, Negatives: {row['negative_target_count']})")
        report_lines.append(f"- **Accuracy**: {row['accuracy']:.4f}" if pd.notnull(row['accuracy']) else "- **Accuracy**: N/A")
        report_lines.append(f"- **Recall**: {row['recall']:.4f}" if pd.notnull(row['recall']) else "- **Recall**: N/A")
        report_lines.append(f"- **Specificity**: {row['specificity']:.4f}" if pd.notnull(row['specificity']) else "- **Specificity**: N/A")
        report_lines.append(f"- **Precision**: {row['precision']:.4f}" if pd.notnull(row['precision']) else "- **Precision**: N/A")
        report_lines.append(f"- **F1 Score**: {row['f1']:.4f}" if pd.notnull(row['f1']) else "- **F1 Score**: N/A")
        report_lines.append("")
        
    with open(reports_dir / "subgroup_analysis.md", "w") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    run_subgroup_analysis()
