FEATURES_ORDER = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_induced_angina", "st_depression", "st_slope",
    "num_major_vessels", "thalassemia"
]

NUMERIC_FEATURES = [
    "age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"
]

CATEGORICAL_FEATURES = [
    "sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg", 
    "exercise_induced_angina", "st_slope", "thalassemia"
]

DISCRETE_FEATURES = [
    "num_major_vessels"
]

TARGET = "target"

CATEGORY_MAPPING = {
    "sex": {0.0: "Female", 1.0: "Male"},
    "chest_pain_type": {1.0: "Typical Angina", 2.0: "Atypical Angina", 3.0: "Non-anginal Pain", 4.0: "Asymptomatic"},
    "fasting_blood_sugar": {0.0: "No", 1.0: "Yes"},
    "resting_ecg": {0.0: "Normal", 1.0: "ST-T Wave Abnormality", 2.0: "Left Ventricular Hypertrophy"},
    "exercise_induced_angina": {0.0: "No", 1.0: "Yes"},
    "st_slope": {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"},
    "thalassemia": {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}
}

TARGET_MAPPING = {
    0: "Absence of Heart Disease",
    1: "Presence of Heart Disease"
}
