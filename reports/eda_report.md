# Exploratory Data Analysis Report

## 1. Dataset Overview
- **Total Samples**: 303
- **Total Features**: 13

## 2. Target Distribution
- **Absence of Heart Disease**: 164
- **Presence of Heart Disease**: 139

## 3. Missingness Analysis
- **num_major_vessels**: 4 missing values
- **thalassemia**: 2 missing values


## 4. Numeric Feature Findings
### age
- Count: 303.0
- Mean: 54.44
- Std: 9.04
- Min: 29.00
- Q1: 48.00
- Median: 56.00
- Q3: 61.00
- Max: 77.00
- IQR: 13.00
- Diagnostic IQR Outliers: 0

### resting_bp
- Count: 303.0
- Mean: 131.69
- Std: 17.60
- Min: 94.00
- Q1: 120.00
- Median: 130.00
- Q3: 140.00
- Max: 200.00
- IQR: 20.00
- Diagnostic IQR Outliers: 9

### cholesterol
- Count: 303.0
- Mean: 246.69
- Std: 51.78
- Min: 126.00
- Q1: 211.00
- Median: 241.00
- Q3: 275.00
- Max: 564.00
- IQR: 64.00
- Diagnostic IQR Outliers: 5

### max_heart_rate
- Count: 303.0
- Mean: 149.61
- Std: 22.88
- Min: 71.00
- Q1: 133.50
- Median: 153.00
- Q3: 166.00
- Max: 202.00
- IQR: 32.50
- Diagnostic IQR Outliers: 1

### st_depression
- Count: 303.0
- Mean: 1.04
- Std: 1.16
- Min: 0.00
- Q1: 0.00
- Median: 0.80
- Q3: 1.60
- Max: 6.20
- IQR: 1.60
- Diagnostic IQR Outliers: 5

## 5. Categorical Feature Findings
### sex
- **Female**: Count = 97, Disease Proportion = 25.77%
- **Male**: Count = 206, Disease Proportion = 55.34%


### chest_pain_type
- **Typical Angina**: Count = 23, Disease Proportion = 30.43%
- **Atypical Angina**: Count = 50, Disease Proportion = 18.00%
- **Non-anginal Pain**: Count = 86, Disease Proportion = 20.93%
- **Asymptomatic**: Count = 144, Disease Proportion = 72.92%


### fasting_blood_sugar
- **No**: Count = 258, Disease Proportion = 45.35%
- **Yes**: Count = 45, Disease Proportion = 48.89%


### resting_ecg
- **Normal**: Count = 151, Disease Proportion = 37.09%
- **ST-T Wave Abnormality**: Count = 4, Disease Proportion = 75.00%
- **Left Ventricular Hypertrophy**: Count = 148, Disease Proportion = 54.05%


### exercise_induced_angina
- **No**: Count = 204, Disease Proportion = 30.88%
- **Yes**: Count = 99, Disease Proportion = 76.77%


### st_slope
- **Upsloping**: Count = 142, Disease Proportion = 25.35%
- **Flat**: Count = 140, Disease Proportion = 65.00%
- **Downsloping**: Count = 21, Disease Proportion = 57.14%


### thalassemia
- **Normal**: Count = 166, Disease Proportion = 22.29%
- **Fixed Defect**: Count = 18, Disease Proportion = 66.67%
- **Reversible Defect**: Count = 117, Disease Proportion = 76.07%
- **Missing**: Count = 2, Disease Proportion = 50.00%


### num_major_vessels
- **0.0**: Count = 176, Disease Proportion = 26.14%
- **1.0**: Count = 65, Disease Proportion = 67.69%
- **2.0**: Count = 38, Disease Proportion = 81.58%
- **3.0**: Count = 20, Disease Proportion = 85.00%
- **Missing**: Count = 4, Disease Proportion = 25.00%


## 6. Feature-Target Associations
The dataset shows an association between age, max heart rate, and certain chest pain types with the presence of heart disease. Nominal category codes were not treated as continuous measurements.

## 7. Potential Data Quality and Range Observations
Some features contain outliers based on the IQR method (e.g., cholesterol), but these represent plausible clinical values and were not removed.

## 8. Preprocessing Implications
Missing values exist in `num_major_vessels` and `thalassemia` and will require imputation. Numeric features show varying scales, suggesting standard scaling may be necessary for scale-sensitive models. Categorical variables require proper encoding.

## 9. EDA Limitations
This analysis is observational. We do not claim causality, diagnosis capability, or clinical validation based on these associations.