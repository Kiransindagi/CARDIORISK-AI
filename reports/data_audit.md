# Data Audit Report

## Dataset Information
- **Source**: UCI Heart Disease Cleveland
- **Shape (Rows, Columns)**: 303, 14
- **Duplicate Count**: 0

## Missing Values Analysis
- **num_major_vessels**: 4 missing values
- **thalassemia**: 2 missing values

## Target Transformation
The original target column ('num') ranges from 0 to 4.
- 0: Absence of heart disease
- 1, 2, 3, 4: Presence of heart disease

After mapping to binary:
- **Class 0 (No Disease)**: 164
- **Class 1 (Disease)**: 139

## Feature Types & Statistics
### age
- Type: float64
- Unique Values: 41
- Min: 29.00, Max: 77.00, Mean: 54.44
### sex
- Type: float64
- Unique Values: 2
- Min: 0.00, Max: 1.00, Mean: 0.68
### chest_pain_type
- Type: float64
- Unique Values: 4
- Min: 1.00, Max: 4.00, Mean: 3.16
### resting_bp
- Type: float64
- Unique Values: 50
- Min: 94.00, Max: 200.00, Mean: 131.69
### cholesterol
- Type: float64
- Unique Values: 152
- Min: 126.00, Max: 564.00, Mean: 246.69
### fasting_blood_sugar
- Type: float64
- Unique Values: 2
- Min: 0.00, Max: 1.00, Mean: 0.15
### resting_ecg
- Type: float64
- Unique Values: 3
- Min: 0.00, Max: 2.00, Mean: 0.99
### max_heart_rate
- Type: float64
- Unique Values: 91
- Min: 71.00, Max: 202.00, Mean: 149.61
### exercise_induced_angina
- Type: float64
- Unique Values: 2
- Min: 0.00, Max: 1.00, Mean: 0.33
### st_depression
- Type: float64
- Unique Values: 40
- Min: 0.00, Max: 6.20, Mean: 1.04
### st_slope
- Type: float64
- Unique Values: 3
- Min: 1.00, Max: 3.00, Mean: 1.60
### num_major_vessels
- Type: float64
- Unique Values: 4
- Min: 0.00, Max: 3.00, Mean: 0.67
### thalassemia
- Type: float64
- Unique Values: 3
- Min: 3.00, Max: 7.00, Mean: 4.73

## Validation Check
- [x] Dataset loads successfully
- [x] Exactly one target is identified
- [x] Expected 13 predictors are present after mapping (Found 13)
- [x] Target conversion is correct
- [x] Missing-value handling strategy is documented (replaced '?' with NaN)
- [x] Class counts are printed
- [x] Duplicate rows are checked
- [x] No target leakage columns exist