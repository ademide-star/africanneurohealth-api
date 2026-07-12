"""
African NeuroHealth Intelligence — FastAPI Backend
Loads your trained .pkl models and serves predictions to the HTML frontend.
Deploy on Render (free tier) at: https://africanneurohealth-api.onrender.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="African NeuroHealth Intelligence API",
    description="Stroke and Dementia risk prediction using trained ML models",
    version="2.1.0"
)

# ── CORS — allow your Vercel site and HF space ──
# NOTE: "*" was removed. Per the CORS spec, a wildcard origin cannot be
# combined with allow_credentials=True — browsers will reject credentialed
# requests against a wildcard response. List every real origin explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://african-neurohealth-dashboard.vercel.app",
        "https://ademideola-african-neurohealth.hf.space",
        "https://neuromatrixbiosystems.com",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──
stroke_model  = None
dementia_model = None


@app.on_event("startup")
def load_models():
    global stroke_model, dementia_model

    stroke_paths = [
        "stroke_REAL_model.pkl",
        "stroke_model.pkl",
        "stroke_pipeline.joblib",
    ]
    for path in stroke_paths:
        if os.path.exists(path):
            try:
                stroke_model = joblib.load(path)
                logger.info(f"✅ Stroke model loaded from {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load stroke model from {path}: {e}")

    dementia_paths = [
        "african_neurohealth_hgb.pkl",
        "best_model_HistGradientBoosting.pkl",
        "alzheimers_pipeline.joblib",
    ]
    for path in dementia_paths:
        if os.path.exists(path):
            try:
                dementia_model = joblib.load(path)
                logger.info(f"✅ Dementia model loaded from {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load dementia model from {path}: {e}")

    if not stroke_model:
        logger.warning("⚠️  Stroke model not loaded — predictions will return demo values")
    if not dementia_model:
        logger.warning("⚠️  Dementia model not loaded — predictions will return demo values")


# ── Health check ──
@app.get("/")
def root():
    return {
        "service": "African NeuroHealth Intelligence API",
        "status": "running",
        "stroke_model":  "loaded" if stroke_model  else "demo mode",
        "dementia_model": "loaded" if dementia_model else "demo mode",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": {
        "stroke":   stroke_model  is not None,
        "dementia": dementia_model is not None
    }}


# ════════════════════════════════════════════
#  STROKE PREDICTION
# ════════════════════════════════════════════

class StrokeInput(BaseModel):
    # Numeric
    age:               float
    avg_glucose_level: float = 100.0
    bmi:               float = 25.0
    stress_level:      float = 0.0
    ptsd:              float = 0.0
    depression_level:  float = 0.0
    diabetes_type:     float = 0.0
    sleep_hours:       float = 7.0
    height:            float = 170.0
    weight:            float = 70.0
    systolic_bp:       float = 120.0
    diastolic_bp:      float = 80.0
    # Categorical
    gender:            str   = "Male"
    ever_married:      str   = "No"
    work_type:         str   = "Private"
    Residence_type:    str   = "Urban"
    smoking_status:    str   = "Never smoked"
    blood_group:       str   = "O+"
    genotype:          str   = "AA"
    # Boolean
    hypertension:      int   = 0
    heart_disease:     int   = 0
    chronic_pain_None:          int = 1
    chronic_pain_Rheumatism:    int = 0
    chronic_pain_Osteoarthritis:int = 0
    chronic_pain_Others:        int = 0
    salt_intake_High:     int = 0
    salt_intake_Little:   int = 0
    salt_intake_Moderate: int = 0
    salt_intake_None:     int = 1
    hypertension_treatment_Drugs:  int = 0
    hypertension_treatment_Herbal: int = 0
    hypertension_treatment_None:   int = 1
    nutritional_lifestyle_Fast_Foods:          int = 0
    nutritional_lifestyle_Homemade_Food:       int = 0
    nutritional_lifestyle_Junk_Food:           int = 0
    nutritional_lifestyle_Local_Bukka:         int = 0
    noise_sources_Block_Industry:  int = 0
    noise_sources_Church:          int = 0
    noise_sources_Club_House:      int = 0
    noise_sources_Generator:       int = 0
    noise_sources_Grinding_Machine:int = 0
    noise_sources_Market:          int = 0
    noise_sources_Mosque:          int = 0
    noise_sources_None:            int = 1
    noise_sources_Welder:          int = 0
    # Location
    country:   Optional[str] = ""
    province:  Optional[str] = ""
    region:    Optional[str] = ""
    ethnicity: Optional[str] = ""


def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def build_stroke_df(data: StrokeInput) -> pd.DataFrame:
    """
    Build the exact DataFrame the stroke model expects.
    Columns/order here match model.feature_names_in_ exactly — do not add,
    remove, or reorder without re-checking feature_names_in_ on the .pkl.
    """
    expected = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
        'smoking_status', 'stress_level', 'ptsd', 'depression_level',
        'diabetes_type', 'sleep_hours',
        'chronic_pain_None', 'chronic_pain_Osteoarthritis',
        'chronic_pain_Others', 'chronic_pain_Rheumatism',
        'salt_intake_High', 'salt_intake_Little',
        'salt_intake_Moderate', 'salt_intake_None',
        'hypertension_treatment_Drugs', 'hypertension_treatment_Herbal',
        'hypertension_treatment_None',
        'nutritional_lifestyle_Fast Foods',
        'nutritional_lifestyle_Homemade Food',
        'nutritional_lifestyle_Junk Food',
        'nutritional_lifestyle_Local Bukka/Street Food',
        'noise_sources_Block-Industry', 'noise_sources_Church',
        'noise_sources_Club-House', 'noise_sources_Generator',
        'noise_sources_Grinding-Machine', 'noise_sources_Market',
        'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder'
    ]

    d = data.dict()
    row = {
        'gender': str(d.get('gender', 'Male') or 'Male'),
        'age': safe_float(d.get('age', 0)),
        'hypertension': int(d.get('hypertension', 0) or 0),
        'heart_disease': int(d.get('heart_disease', 0) or 0),
        'ever_married': str(d.get('ever_married', 'No') or 'No'),
        'work_type': str(d.get('work_type', 'Private') or 'Private'),
        'Residence_type': str(d.get('Residence_type', 'Urban') or 'Urban'),
        'avg_glucose_level': safe_float(d.get('avg_glucose_level', 100)),
        'bmi': safe_float(d.get('bmi', 25)),
        'smoking_status': str(d.get('smoking_status', 'Never smoked') or 'Never smoked'),
        'stress_level': safe_float(d.get('stress_level', 0)),
        'ptsd': safe_float(d.get('ptsd', 0)),
        'depression_level': safe_float(d.get('depression_level', 0)),
        'diabetes_type': safe_float(d.get('diabetes_type', 0)),
        'sleep_hours': safe_float(d.get('sleep_hours', 7)),
        'chronic_pain_None': d.get('chronic_pain_None', 1),
        'chronic_pain_Osteoarthritis': d.get('chronic_pain_Osteoarthritis', 0),
        'chronic_pain_Others': d.get('chronic_pain_Others', 0),
        'chronic_pain_Rheumatism': d.get('chronic_pain_Rheumatism', 0),
        'salt_intake_High': d.get('salt_intake_High', 0),
        'salt_intake_Little': d.get('salt_intake_Little', 0),
        'salt_intake_Moderate': d.get('salt_intake_Moderate', 0),
        'salt_intake_None': d.get('salt_intake_None', 1),
        'hypertension_treatment_Drugs': d.get('hypertension_treatment_Drugs', 0),
        'hypertension_treatment_Herbal': d.get('hypertension_treatment_Herbal', 0),
        'hypertension_treatment_None': d.get('hypertension_treatment_None', 1),
        'nutritional_lifestyle_Fast Foods': d.get('nutritional_lifestyle_Fast_Foods', 0),
        'nutritional_lifestyle_Homemade Food': d.get('nutritional_lifestyle_Homemade_Food', 0),
        'nutritional_lifestyle_Junk Food': d.get('nutritional_lifestyle_Junk_Food', 0),
        'nutritional_lifestyle_Local Bukka/Street Food': d.get('nutritional_lifestyle_Local_Bukka', 0),
        'noise_sources_Block-Industry': d.get('noise_sources_Block_Industry', 0),
        'noise_sources_Church': d.get('noise_sources_Church', 0),
        'noise_sources_Club-House': d.get('noise_sources_Club_House', 0),
        'noise_sources_Generator': d.get('noise_sources_Generator', 0),
        'noise_sources_Grinding-Machine': d.get('noise_sources_Grinding_Machine', 0),
        'noise_sources_Market': d.get('noise_sources_Market', 0),
        'noise_sources_Mosque': d.get('noise_sources_Mosque', 0),
        'noise_sources_None': d.get('noise_sources_None', 1),
        'noise_sources_Welder': d.get('noise_sources_Welder', 0),
    }
    return pd.DataFrame([row])[expected]


@app.post("/predict/stroke")
def predict_stroke(data: StrokeInput):
    try:
        df = build_stroke_df(data)

        if stroke_model is None:
            risk_score = _demo_stroke_score(data)
        else:
            model = stroke_model.get('model') if isinstance(stroke_model, dict) else stroke_model
            proba = model.predict_proba(df)[0]
            risk_score = float(proba[1])

        risk_pct   = round(risk_score * 100, 1)
        risk_level = "HIGH" if risk_score > 0.65 else "MEDIUM" if risk_score > 0.30 else "LOW"

        factors = _stroke_risk_factors(data)
        recos   = _stroke_recommendations(data, risk_level)

        return {
            "risk_score":  risk_score,
            "risk_pct":    risk_pct,
            "risk_level":  risk_level,
            "risk_factors": factors,
            "recommendations": recos,
            "model_used":  "trained" if stroke_model else "demo",
            "timestamp":   datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stroke prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _demo_stroke_score(d: StrokeInput) -> float:
    score = 0.0
    if d.age > 60:              score += 0.20
    elif d.age > 45:            score += 0.10
    if d.hypertension:          score += 0.18
    if d.heart_disease:         score += 0.15
    if d.avg_glucose_level > 200: score += 0.12
    if d.systolic_bp > 140:     score += 0.10
    if d.smoking_status == "Smokes": score += 0.10
    if d.bmi > 30:              score += 0.08
    if d.stress_level >= 2:     score += 0.06
    if d.ptsd:                  score += 0.05
    if d.salt_intake_High:      score += 0.04
    return min(0.97, score)


def _stroke_risk_factors(d: StrokeInput) -> list:
    f = []
    if d.age > 60:            f.append("Age above 60 years")
    if d.hypertension:        f.append("Hypertension")
    if d.heart_disease:       f.append("Heart disease")
    if d.avg_glucose_level > 200: f.append(f"High blood glucose ({d.avg_glucose_level} mg/dL)")
    if d.systolic_bp > 140:   f.append(f"Elevated systolic BP ({d.systolic_bp} mmHg)")
    if d.smoking_status == "Smokes": f.append("Current smoker")
    if d.bmi > 30:            f.append(f"Obesity (BMI {d.bmi:.1f})")
    if d.stress_level >= 2:   f.append("High stress level")
    if d.ptsd:                f.append("PTSD")
    if d.salt_intake_High:    f.append("High salt intake")
    return f if f else ["No major risk factors identified"]


def _stroke_recommendations(d: StrokeInput, level: str) -> list:
    r = []
    if d.hypertension:        r.append("Strict BP control — target <130/80 mmHg")
    if d.smoking_status == "Smokes": r.append("Smoking cessation programme")
    if d.bmi > 30:            r.append("Weight management — target BMI <25")
    if d.avg_glucose_level > 126: r.append("HbA1c monitoring and glycaemic control")
    r.append("Mediterranean-style diet — vegetables, fish, olive oil, low salt")
    r.append("At least 150 min moderate physical activity per week")
    r.append("Annual neurological health check-up")
    if level == "HIGH": r.append("Immediate consultation with a physician or neurologist")
    return r


# ════════════════════════════════════════════
#  DEMENTIA PREDICTION
# ════════════════════════════════════════════

class DementiaInput(BaseModel):
    # Numeric (exact names from prepare_alzheimers_input_numeric)
    Age:                     float
    BMI:                     float = 25.0
    EducationLevel:          float = 12.0
    AlcoholConsumption:      float = 0.0
    PhysicalActivity:        float = 3.0
    DietQuality:              float = 5.0
    SleepQuality:            float = 6.0
    SystolicBP:              float = 120.0
    DiastolicBP:              float = 80.0
    CholesterolTotal:        float = 200.0
    CholesterolLDL:          float = 120.0
    CholesterolHDL:          float = 50.0
    CholesterolTriglycerides:float = 150.0
    FunctionalAssessment:    float = 8.0
    ADL:                     float = 8.0
    HeadInjury:              float = 0.0
    MMSE:                    float = 27.0
    Height:                  float = 170.0
    Weight:                  float = 70.0
    PollutionScore:          float = 20.0
    Ethnicity:               float = 0.0
    Country:                 float = 0.0
    Province_Option:         float = 0.0
    MemoryScore:             float = 0.0
    CustomStressScore:       float = 0.0
    # Categorical
    Gender:                  str = "Male"
    Smoking:                 str = "No"
    FamilyHistoryAlzheimers: str = "No"
    CardiovascularDisease:   str = "No"
    Diabetes:                str = "No"
    Depression:              str = "No"
    Hypertension:            str = "No"
    BehavioralProblems:      str = "No"
    Genotype:                str = "AA"
    BloodGroup:              str = "O+"
    # Boolean
    Confusion:                  int = 0
    Disorientation:             int = 0
    PersonalityChanges:         int = 0
    DifficultyCompletingTasks:  int = 0
    Forgetfulness:              int = 0
    MemoryComplaints:           int = 0
    PollutionCategoryLow:       int = 1
    PollutionCategoryModerate:  int = 0
    PollutionCategoryHigh:      int = 0
    # Extra
    country_name:  Optional[str] = ""
    province_name: Optional[str] = ""
    region_name:   Optional[str] = ""
    ethnicity_name:Optional[str] = ""


def build_dementia_df(data: DementiaInput) -> pd.DataFrame:
    """
    Build the exact DataFrame the dementia model expects.
    Columns/order here match model.feature_names_in_ exactly — do not add,
    remove, or reorder without re-checking feature_names_in_ on the .pkl.
    """
    expected = [
        'Age', 'Gender', 'EducationLevel', 'BMI', 'Smoking',
        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
        'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
        'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
        'Confusion', 'Disorientation', 'PersonalityChanges',
        'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    numeric_cols = [
        'Age', 'BMI', 'EducationLevel', 'AlcoholConsumption',
        'PhysicalActivity', 'DietQuality', 'SleepQuality',
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
        'FunctionalAssessment', 'ADL', 'HeadInjury', 'MMSE'
    ]
    categorical_cols = [
        'Gender', 'Smoking', 'FamilyHistoryAlzheimers',
        'CardiovascularDisease', 'Diabetes', 'Depression',
        'Hypertension', 'BehavioralProblems'
    ]
    boolean_cols = [
        'Confusion', 'Disorientation', 'PersonalityChanges',
        'DifficultyCompletingTasks', 'Forgetfulness', 'MemoryComplaints'
    ]

    d = data.dict()
    row = {}
    for col in numeric_cols:
        row[col] = safe_float(d.get(col, 0))
    for col in categorical_cols:
        row[col] = str(d.get(col, "No") or "No")
    for col in boolean_cols:
        row[col] = int(d.get(col, 0) or 0)

    return pd.DataFrame([row])[expected]


@app.post("/predict/dementia")
def predict_dementia(data: DementiaInput):
    try:
        df = build_dementia_df(data)

        if dementia_model is None:
            risk_score = _demo_dementia_score(data)
        else:
            model = dementia_model.get('model') if isinstance(dementia_model, dict) else dementia_model
            # Handle ensemble dict from Streamlit app, if the pkl was saved that way
            if isinstance(model, dict):
                probas = []
                for name, m in model.items():
                    try:
                        p = m.predict_proba(df)[0][1]
                        probas.append(p)
                    except Exception:
                        pass
                risk_score = float(np.mean(probas)) if probas else _demo_dementia_score(data)
            else:
                risk_score = float(model.predict_proba(df)[0][1])

        risk_pct   = round(risk_score * 100, 1)
        risk_level = "HIGH" if risk_score > 0.60 else "MEDIUM" if risk_score > 0.30 else "LOW"

        # MMSE interpretation
        mmse = data.MMSE
        if mmse >= 27:   mmse_label = "Normal cognition"
        elif mmse >= 24: mmse_label = "Mild cognitive impairment"
        elif mmse >= 19: mmse_label = "Moderate cognitive impairment"
        else:            mmse_label = "Severe cognitive impairment — urgent referral recommended"

        factors = _dementia_risk_factors(data)
        recos   = _dementia_recommendations(data, risk_level)

        return {
            "risk_score":  risk_score,
            "risk_pct":    risk_pct,
            "risk_level":  risk_level,
            "mmse":        mmse,
            "mmse_label":  mmse_label,
            "risk_factors": factors,
            "recommendations": recos,
            "model_used":  "trained" if dementia_model else "demo",
            "timestamp":   datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Dementia prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _demo_dementia_score(d: DementiaInput) -> float:
    score = 0.0
    if d.Age > 75:              score += 0.22
    elif d.Age > 65:            score += 0.14
    if d.MMSE < 24:             score += 0.22
    elif d.MMSE < 27:           score += 0.10
    if d.FamilyHistoryAlzheimers == "Yes": score += 0.16
    if d.Depression == "Yes":   score += 0.10
    if d.CardiovascularDisease == "Yes": score += 0.09
    if d.Hypertension == "Yes": score += 0.07
    if d.HeadInjury > 0:        score += 0.08
    if d.PhysicalActivity < 2:  score += 0.06
    if d.DietQuality < 4:       score += 0.05
    if d.SleepQuality < 4:      score += 0.05
    return min(0.97, score)


def _dementia_risk_factors(d: DementiaInput) -> list:
    f = []
    if d.Age > 65:             f.append(f"Age above 65 years ({d.Age:.0f})")
    if d.MMSE < 24:            f.append(f"MMSE score {d.MMSE:.0f}/30 — cognitive impairment")
    if d.FamilyHistoryAlzheimers == "Yes": f.append("Family history of Alzheimer's")
    if d.Depression == "Yes":  f.append("Depression")
    if d.CardiovascularDisease == "Yes": f.append("Cardiovascular disease")
    if d.Hypertension == "Yes":f.append("Hypertension")
    if d.HeadInjury > 0:       f.append("History of head injury")
    if d.CholesterolTotal > 240: f.append(f"High cholesterol ({d.CholesterolTotal:.0f} mg/dL)")
    if d.PhysicalActivity < 2: f.append("Physical inactivity")
    if d.SleepQuality < 4:     f.append("Poor sleep quality")
    return f if f else ["No major risk factors identified"]


def _dementia_recommendations(d: DementiaInput, level: str) -> list:
    r = []
    if d.FamilyHistoryAlzheimers == "Yes": r.append("Genetic counselling recommended")
    if d.Depression == "Yes":   r.append("Mental health support and therapy")
    if d.Hypertension == "Yes": r.append("Blood pressure management — target <130/80")
    if d.PhysicalActivity < 2:  r.append("150 min/week aerobic exercise")
    if d.DietQuality < 5:       r.append("Mediterranean or MIND diet")
    r.append("Omega-3 fatty acids and antioxidant-rich foods")
    r.append("Social engagement and cognitively stimulating activities")
    r.append("Regular cognitive screening every 6–12 months")
    if level == "HIGH":          r.append("Urgent referral to a neurologist")
    return r


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
