# Model Card: CardioRisk AI

## Model Overview
- **Final Architecture**: LogisticRegression_Calibrated
- **Preprocessing**: Imputation, Scaling, OneHotEncoding
- **Imbalance Handling**: class_weight=balanced natively in LR
- **Calibration Status**: Uncalibrated (calibration worsened Brier score)
- **Decision Threshold**: 0.44000000000000006
- **Model Version**: 1.0.0

## Intended Use
Educational and portfolio demonstration of structured-data ML, explainability, threshold optimization, API deployment, and ML engineering.

## Not Intended For
- Medical diagnosis
- Emergency decisions
- Treatment recommendations
- Clinical screening
- Autonomous healthcare decisions
- Replacement of healthcare professionals

## Dataset
- **Source**: UCI Cleveland Heart Disease dataset
- **Size**: 303 observations
- **Features**: 13 predictive features
- **Target**: Binary target (0: Absence, 1: Presence originally 1-4)

## Evaluation Strategy
- **Split**: 80/20 stratified development/test split
- **Development Samples**: 242
- **Test Samples**: 61
- **CV Strategy**: 5-fold Stratified CV for predictions, Repeated for baseline metrics
- **Threshold Selection**: 'recall >= 0.85 then max F1' on development out-of-fold probabilities

## Performance
### Development Validation Results (Out-of-Fold)
- **Recall**: 0.8559
- **Specificity**: 0.8397
- **Precision**: 0.8190
- **F1**: 0.8370

### Final Held-Out Test Results
- **Accuracy**: 0.8852
- **Recall**: 0.9643
- **Specificity**: 0.8182
- **Precision**: 0.8182
- **F1**: 0.8852
- **ROC-AUC**: 0.9556

## Explainability
- **Method**: SHAP (KernelExplainer)
- **Global Explanations**: Summary and bar plots show mean absolute SHAP values.
- **Local Explanations**: Breakdown of individual predictions into risk-increasing and decreasing factors.
- **Limitations**: SHAP measures attribution, not causality.

## Limitations
Small historical dataset, limited representativeness, possible dataset shift, limited subgroup analysis, no external validation, no clinical workflow validation. Model probability is not a diagnosis.