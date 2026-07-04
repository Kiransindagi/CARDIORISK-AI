# SHAP Explainability Report
## Overview
SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance.
## Global Explainability
Global explanations show the overall impact of features across the evaluated dataset.
Based on mean absolute SHAP values, the top features influencing the model are observed in the generated CSV and plots.
## Local Explainability
Local explanations break down individual predictions into risk-increasing and risk-decreasing factors. A reusable function was implemented to support single-patient inference API calls.
## Limitations
SHAP measures feature attribution, not causality. High SHAP values do not prove clinical correctness.