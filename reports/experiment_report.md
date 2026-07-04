# Experiment Report

1. **Data Acquisition**: UCI Cleveland Heart Disease dataset downloaded.
2. **Data Audit**: Identified 4 missing in `num_major_vessels` and 2 in `thalassemia`.
3. **EDA**: Explored distributions and associations between features and target.
4. **Development/Test Split**: 80/20 stratified split (seed 42).
5. **Preprocessing**: Leakage-safe pipelines established.
6. **Baseline**: Logistic Regression established initial baseline.
7. **Imbalance Strategy Comparison**: Class weighting selected over SMOTE to maximize recall.
8. **Model Benchmarking**: Logistic Regression outperformed Random Forest and HistGradientBoosting.
9. **Calibration Comparison**: Uncalibrated vs Sigmoid evaluated.
10. **Calibration Methodology Audit**: Reverted to uncalibrated Logistic Regression as it demonstrated a better Brier score (0.119 vs 0.123).
11. **Threshold Optimization**: Threshold of 0.42 selected from development OOF predictions.
12. **Final Model Selection**: Uncalibrated Logistic Regression with balanced class weights locked.
13. **Held-Out Evaluation**: Achieved a Recall of 0.9643 and F1 of 0.8852 on the 61-sample test set.
14. **SHAP Explainability**: KernelExplainer implemented for global and local insights.
15. **Clinical Plausibility**: Reviewed feature dependencies against clinical basics.
16. **Subgroup Analysis**: Explored performance across age and sex groups, noting small sample instability.