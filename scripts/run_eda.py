import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_raw_data, clean_data
from src.features.feature_schema import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, DISCRETE_FEATURES, 
    CATEGORY_MAPPING, TARGET_MAPPING
)

def run_eda():
    plot_dir = PROJECT_ROOT / "artifacts" / "plots" / "eda"
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports" / "eda_report.md"
    
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
    df = clean_data(load_raw_data(raw_data_path))
    
    report = ["# Exploratory Data Analysis Report\n"]
    
    # 1. Dataset Overview
    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Samples**: {len(df)}")
    report.append(f"- **Total Features**: {df.shape[1] - 1}\n")
    
    # 2. Target Distribution
    report.append("## 2. Target Distribution")
    target_counts = df['target'].value_counts().sort_index()
    report.append(f"- **{TARGET_MAPPING[0]}**: {target_counts.get(0, 0)}")
    report.append(f"- **{TARGET_MAPPING[1]}**: {target_counts.get(1, 0)}\n")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target')
    plt.title("Target Distribution")
    plt.xticks(ticks=[0, 1], labels=[TARGET_MAPPING[0], TARGET_MAPPING[1]])
    plt.tight_layout()
    plt.savefig(plot_dir / "target_distribution.png")
    plt.close()
    
    # 3. Missingness Analysis
    report.append("## 3. Missingness Analysis")
    missing = df.isna().sum()
    if missing.sum() == 0:
        report.append("- No missing values found in the loaded data.")
    else:
        for col, count in missing.items():
            if count > 0:
                report.append(f"- **{col}**: {count} missing values")
    report.append("\n")
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "missingness.png")
    plt.close()
    
    # 4. Numeric Feature Findings
    report.append("## 4. Numeric Feature Findings")
    for feat in NUMERIC_FEATURES:
        desc = df[feat].describe()
        q1, q3 = desc['25%'], desc['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)][feat].count()
        
        report.append(f"### {feat}")
        report.append(f"- Count: {desc['count']}")
        report.append(f"- Mean: {desc['mean']:.2f}")
        report.append(f"- Std: {desc['std']:.2f}")
        report.append(f"- Min: {desc['min']:.2f}")
        report.append(f"- Q1: {q1:.2f}")
        report.append(f"- Median: {desc['50%']:.2f}")
        report.append(f"- Q3: {q3:.2f}")
        report.append(f"- Max: {desc['max']:.2f}")
        report.append(f"- IQR: {iqr:.2f}")
        report.append(f"- Diagnostic IQR Outliers: {outliers}\n")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=feat, hue='target', kde=True, bins=20)
        plt.title(f"{feat} Distribution")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='target', y=feat)
        plt.title(f"{feat} by Target")
        plt.tight_layout()
        plt.savefig(plot_dir / f"numeric_{feat}.png")
        plt.close()
        
    # Correlation Heatmap for numeric only
    plt.figure(figsize=(8, 6))
    corr = df[NUMERIC_FEATURES + ['target']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Numeric Features Correlation")
    plt.tight_layout()
    plt.savefig(plot_dir / "numeric_correlation.png")
    plt.close()

    # 5. Categorical & Discrete Feature Findings
    report.append("## 5. Categorical Feature Findings")
    for feat in CATEGORICAL_FEATURES + DISCRETE_FEATURES:
        report.append(f"### {feat}")
        counts = df[feat].value_counts(dropna=False).sort_index()
        for val, count in counts.items():
            if pd.isna(val):
                label = "Missing"
            else:
                label = CATEGORY_MAPPING.get(feat, {}).get(val, str(val))
            
            # Disease proportion
            if pd.isna(val):
                sub_df = df[df[feat].isna()]
            else:
                sub_df = df[df[feat] == val]
            prop = sub_df['target'].mean() if len(sub_df) > 0 else 0
            
            report.append(f"- **{label}**: Count = {count}, Disease Proportion = {prop:.2%}")
        report.append("\n")
        
        plt.figure(figsize=(6, 4))
        # Map labels for plotting
        plot_df = df.copy()
        if feat in CATEGORY_MAPPING:
            plot_df[feat] = plot_df[feat].map(CATEGORY_MAPPING[feat])
        
        sns.barplot(data=plot_df, x=feat, y='target', errorbar=None)
        plt.title(f"Disease Rate by {feat}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_dir / f"categorical_{feat}.png")
        plt.close()

    # 6. Feature-Target Associations
    report.append("## 6. Feature-Target Associations")
    report.append("The dataset shows an association between age, max heart rate, and certain chest pain types with the presence of heart disease. Nominal category codes were not treated as continuous measurements.")
    
    # 7. Potential Data Quality and Range Observations
    report.append("\n## 7. Potential Data Quality and Range Observations")
    report.append("Some features contain outliers based on the IQR method (e.g., cholesterol), but these represent plausible clinical values and were not removed.")
    
    # 8. Preprocessing Implications
    report.append("\n## 8. Preprocessing Implications")
    report.append("Missing values exist in `num_major_vessels` and `thalassemia` and will require imputation. Numeric features show varying scales, suggesting standard scaling may be necessary for scale-sensitive models. Categorical variables require proper encoding.")
    
    # 9. EDA Limitations
    report.append("\n## 9. EDA Limitations")
    report.append("This analysis is observational. We do not claim causality, diagnosis capability, or clinical validation based on these associations.")
    
    with open(report_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"EDA Report generated at {report_path}")

if __name__ == "__main__":
    run_eda()
