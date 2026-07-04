from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional

class PatientInput(BaseModel):
    age: float = Field(..., description="Age in years", ge=18, le=100)
    sex: str = Field(..., description="Recorded sex")
    chest_pain_type: str = Field(..., description="Chest pain type")
    resting_bp: float = Field(..., description="Resting blood pressure (in mm Hg)", ge=50, le=250)
    cholesterol: float = Field(..., description="Serum cholesterol in mg/dl", ge=50, le=600)
    fasting_blood_sugar: str = Field(..., description="Fasting blood sugar > 120 mg/dl")
    resting_ecg: str = Field(..., description="Resting electrocardiographic results")
    max_heart_rate: float = Field(..., description="Maximum heart rate achieved", ge=50, le=220)
    exercise_induced_angina: str = Field(..., description="Exercise induced angina")
    st_depression: float = Field(..., description="ST depression induced by exercise relative to rest", ge=-2.0, le=10.0)
    st_slope: str = Field(..., description="The slope of the peak exercise ST segment")
    num_major_vessels: float = Field(..., description="Number of major vessels (0-3) colored by flourosopy", ge=0, le=3)
    thalassemia: str = Field(..., description="Thalassemia status")

    @validator('sex')
    def valid_sex(cls, v):
        allowed = ["Male", "Female"]
        if v not in allowed:
            raise ValueError(f"sex must be one of {allowed}")
        return v
        
    @validator('chest_pain_type')
    def valid_cp(cls, v):
        allowed = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        if v not in allowed:
            raise ValueError(f"chest_pain_type must be one of {allowed}")
        return v
        
    @validator('fasting_blood_sugar')
    def valid_fbs(cls, v):
        allowed = ["Yes", "No"]
        if v not in allowed:
            raise ValueError(f"fasting_blood_sugar must be one of {allowed}")
        return v

    @validator('resting_ecg')
    def valid_ecg(cls, v):
        allowed = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        if v not in allowed:
            raise ValueError(f"resting_ecg must be one of {allowed}")
        return v
        
    @validator('exercise_induced_angina')
    def valid_angina(cls, v):
        allowed = ["Yes", "No"]
        if v not in allowed:
            raise ValueError(f"exercise_induced_angina must be one of {allowed}")
        return v
        
    @validator('st_slope')
    def valid_slope(cls, v):
        allowed = ["Upsloping", "Flat", "Downsloping"]
        if v not in allowed:
            raise ValueError(f"st_slope must be one of {allowed}")
        return v
        
    @validator('thalassemia')
    def valid_thal(cls, v):
        allowed = ["Normal", "Fixed Defect", "Reversible Defect"]
        if v not in allowed:
            raise ValueError(f"thalassemia must be one of {allowed}")
        return v

class ExplanationFactor(BaseModel):
    feature: str
    label: str
    value: float | str
    contribution: float
    direction: str

class Explanation(BaseModel):
    risk_increasing_factors: List[ExplanationFactor]
    risk_decreasing_factors: List[ExplanationFactor]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    risk_probability: float
    decision_threshold: float
    risk_band: str
    explanation: Explanation
    version: str
    disclaimer: str
