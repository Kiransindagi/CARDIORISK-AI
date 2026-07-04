# Baseline Model Report
## Architecture
Logistic Regression with no imbalance treatment.
## Validation Method
5-fold Repeated Stratified Cross-Validation (5 repeats) on Development set.
## Results
- **Accuracy**: 0.8264 ± 0.0497
- **Balanced Accuracy**: 0.8216 ± 0.0493
- **Precision**: 0.8465 ± 0.0784
- **Recall/Sensitivity**: 0.7671 ± 0.0683
- **Specificity**: 0.8761 ± 0.0723
- **F1 Score**: 0.8020 ± 0.0555
- **ROC-AUC**: 0.8987 ± 0.0400
- **PR-AUC**: 0.8924 ± 0.0461
## Implications
This baseline demonstrates the fundamental predictability of the dataset. Recall is of particular interest as false negatives are costly in medical contexts.
## Limitations
This is a baseline model and does not claim medical diagnostic capability. Performance is evaluated on development data only.