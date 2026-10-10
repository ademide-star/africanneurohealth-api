"""
African NeuroHealth Intelligence — FastAPI Backend
Loads your trained .pkl models and serves predictions to the HTML frontend.
Deploy on Render (free tier) at: https://africanneurohealth-api-jhke.onrender.com
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import logging
import re
import json
import uuid
import time
import hmac
import hashlib
import base64
import secrets
import urllib.request
import urllib.error
from urllib.parse import urlencode
from datetime import datetime, timezone

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
        "https://www.neuromatrixbiosystems.com",
        "https://neurohealth.neuromatrixbiosystems.com",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
    ],
    # Any other subdomain of neuromatrixbiosystems.com is allowed too
    allow_origin_regex=r"https://([a-z0-9-]+\.)?neuromatrixbiosystems\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──
stroke_model  = None
dementia_model = None
cognitive_symptom_bundle = None


@app.on_event("startup")
def load_models():
    global stroke_model, dementia_model, cognitive_symptom_bundle

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

    # Load the cognitive-symptom model independently. Failure to load this new
    # bundle must not prevent the existing stroke/dementia endpoints from starting.
    cognitive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "african_cognitive_symptom_rf_platt.joblib")
    if os.path.exists(cognitive_path):
        try:
            cognitive_symptom_bundle = joblib.load(cognitive_path)
            logger.info("Cognitive-symptom model loaded from %s", cognitive_path)
        except Exception as e:
            cognitive_symptom_bundle = None
            logger.exception("Could not load cognitive-symptom model: %s", e)
    else:
        logger.warning("Cognitive-symptom model not found at %s", cognitive_path)


# ── Health check ──
@app.get("/")
def root():
    return {
        "service": "African NeuroHealth Intelligence API",
        "status": "running",
        "stroke_model":  "loaded" if stroke_model  else "demo mode",
        "dementia_model": "loaded" if dementia_model else "demo mode",
        "cognitive_symptom_model": "loaded" if cognitive_symptom_bundle else "not loaded",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": {
        "stroke":   stroke_model  is not None,
        "dementia": dementia_model is not None,
        "cognitive_symptoms": cognitive_symptom_bundle is not None
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

    # Exact codes confirmed from the trained LabelEncoders (le.classes_) on the
    # real training CSV — not guesses. Do not change without re-verifying
    # against the training data.
    def encode(value, mapping, default_key):
        key = str(value).strip().lower().replace(' ', '_')
        return mapping.get(key, mapping[default_key])

    gender_map = {'female': 0, 'male': 1}
    ever_married_map = {'no': 0, 'yes': 1, 'married': 1, 'divorced': 1, 'widowed': 1, 'single': 0}
    # NOTE: the training data never contained a "Never_worked" case, so there
    # is no learned code for it. Mapping it to 'children' (both represent a
    # non-working population) as the closest available proxy.
    work_type_map = {
        'govt_job': 0, 'private': 1, 'self-employed': 2,
        'children': 3, 'child': 3, 'never_worked': 3,
    }
    residence_type_map = {'rural': 0, 'urban': 1}
    smoking_status_map = {
        'unknown': 0, 'formerly_smoked': 1, 'never_smoked': 2, 'smokes': 3,
    }

    d = data.dict()
    row = {
        'gender': encode(d.get('gender', 'Male'), gender_map, 'male'),
        'age': safe_float(d.get('age', 0)),
        'hypertension': int(d.get('hypertension', 0) or 0),
        'heart_disease': int(d.get('heart_disease', 0) or 0),
        'ever_married': encode(d.get('ever_married', 'No'), ever_married_map, 'no'),
        'work_type': encode(d.get('work_type', 'Private'), work_type_map, 'private'),
        'Residence_type': encode(d.get('Residence_type', 'Urban'), residence_type_map, 'urban'),
        'avg_glucose_level': safe_float(d.get('avg_glucose_level', 100)),
        'bmi': safe_float(d.get('bmi', 25)),
        'smoking_status': encode(d.get('smoking_status', 'Never smoked'), smoking_status_map, 'never_smoked'),
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
        'FunctionalAssessment', 'ADL', 'MMSE'
    ]
    # These read as Yes/No or Male/Female in the UI, but — same pattern as
    # stroke's gender/ever_married — the model expects them pre-encoded 0/1,
    # not as strings, since they were label-encoded (not one-hot) at training time.
    binary_cols = [
        'Gender', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'Hypertension', 'BehavioralProblems'
    ]
    boolean_cols = [
        'Confusion', 'Disorientation', 'PersonalityChanges',
        'DifficultyCompletingTasks', 'Forgetfulness', 'MemoryComplaints'
    ]

    def to_binary(v):
        s = str(v).strip().lower()
        return 1 if s in ('yes', 'male', 'true', '1') else 0

    d = data.dict()
    row = {}
    for col in numeric_cols:
        row[col] = safe_float(d.get(col, 0))
    for col in binary_cols:
        row[col] = to_binary(d.get(col, "No"))
    for col in boolean_cols:
        row[col] = int(d.get(col, 0) or 0)

    # Smoking: the model was trained on a BINARY smoking feature (only 2
    # unique values seen), but the UI collects 3 levels (never/formerly/
    # smokes). Collapsing: treat any smoking history (past or present) as 1.
    # If you'd rather only count *current* smokers as 1, change the check to
    # `== 'smokes'` only.
    smoking_raw = str(d.get('Smoking', 'No') or 'No').strip().lower()
    row['Smoking'] = 1 if smoking_raw in ('smokes', 'formerly smoked', 'formerly_smoked', 'yes', '1') else 0

    # HeadInjury: the model was trained on a BINARY head-injury feature, but
    # the UI collects a type (None/Accident/Violence) or a count. Collapsing
    # to Yes/No — any recorded injury type/count > 0 becomes 1.
    head_raw = d.get('HeadInjury', 0)
    try:
        row['HeadInjury'] = 1 if float(head_raw) > 0 else 0
    except (ValueError, TypeError):
        row['HeadInjury'] = 0 if str(head_raw).strip().lower() in ('none', 'no', '') else 1

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


# ════════════════════════════════════════════
#  SELF-REPORTED COGNITIVE-SYMPTOM SCREENING
#  Separate endpoint: existing stroke and dementia routes remain unchanged.
# ════════════════════════════════════════════

COGNITIVE_DEFAULTS = {
    "AGE": 55,
    "SEX": "M",
    "EDU- L": "SECONDARY",
    "S- BP": 120,
    "D-BP": 80,
    "DIABETES": "NO",
    "HYPERTENSION": "NO",
    "HEART- D": "NO",
    "STRESS- L": "M",
    "SLEEP HOURS": 7,
    "CHRONIC- PAIN": "NO",
    "NUTRITION SCORE": 6,
    "HEART RATE": 72,
    "POLLUTION": "L",
    "OCCUPATION HAZZARD": "NO",
    "ACCESS TO HEALTH CARE": "YES",
    "SMOKING STATUS (CORRECTED)": "Non-smoker / not reported",
    "HYPERTENSION TREATMENT (CORRECTED)": "Not applicable",
    "FRUIT_INTAKE": 2,
    "VEGETABLE_INTAKE": 2,
    "HYDRATION_LITERS": 2.0,
    "LIFESTYLE_CHOICES": "Homemade Food",
    "Total_Salt Intake Score": 8,
    "ZONE": "Ijebu Zone",
}

COGNITIVE_NUMERIC_FIELDS = [
    "AGE", "S- BP", "D-BP", "SLEEP HOURS", "NUTRITION SCORE",
    "HEART RATE", "HYDRATION_LITERS", "Total_Salt Intake Score"
]

@app.post("/predict/cognitive-symptoms")
def predict_cognitive_symptoms(payload: dict):
    """Estimate self-reported cognitive-symptom likelihood, not dementia diagnosis."""
    if cognitive_symptom_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Cognitive-symptom model is not loaded. Confirm the model file is deployed under models/.",
        )

    try:
        predictors = list(cognitive_symptom_bundle["predictors"])
        values = dict(COGNITIVE_DEFAULTS)
        # Accept only fields known to this model; missing fields use documented defaults.
        for field in predictors:
            if field in payload and payload[field] is not None and payload[field] != "":
                values[field] = payload[field]

        # Friendly aliases from the HTML form/API clients.
        sex = str(values["SEX"]).strip().lower()
        values["SEX"] = "F" if sex in ("f", "female", "woman") else "M" if sex in ("m", "male", "man") else values["SEX"]
        for field in ("DIABETES", "HYPERTENSION", "HEART- D", "OCCUPATION HAZZARD", "ACCESS TO HEALTH CARE"):
            val = str(values[field]).strip().upper()
            values[field] = "YES" if val in ("YES", "Y", "TRUE", "1") else "NO" if val in ("NO", "N", "FALSE", "0") else values[field]
        for field in COGNITIVE_NUMERIC_FIELDS:
            try:
                values[field] = float(values[field])
            except (TypeError, ValueError):
                values[field] = float(COGNITIVE_DEFAULTS[field])
        for field in ("FRUIT_INTAKE", "VEGETABLE_INTAKE"):
            try:
                values[field] = int(float(values[field]))
            except (TypeError, ValueError):
                values[field] = COGNITIVE_DEFAULTS[field]

        X = pd.DataFrame([{field: values[field] for field in predictors}], columns=predictors)
        raw_probability = cognitive_symptom_bundle["model"].predict_proba(X)[:, 1]
        calibrated_probability = cognitive_symptom_bundle["calibrator"].predict_proba(
            raw_probability.reshape(-1, 1)
        )[0, 1]
        risk_score = float(calibrated_probability)
        risk_pct = round(risk_score * 100, 1)
        risk_level = "LOW" if risk_pct < 30 else "MEDIUM" if risk_pct < 60 else "HIGH"

        return {
            "risk_score": round(risk_score, 6),
            "risk_pct": risk_pct,
            "cognitive_symptom_risk_pct": risk_pct,
            "risk_level": risk_level,
            "risk_factors": [],
            "recommendations": [
                "If memory or thinking concerns persist, arrange an assessment with a qualified healthcare professional.",
                "Continue monitoring blood pressure, sleep, nutrition, and other relevant health factors.",
            ],
            "model_used": "trained",
            "model_name": "African Cognitive Symptom Risk Model",
            "note": (
                "Screening estimate for self-reported cognitive symptoms. It is not a diagnosis of dementia or Alzheimer's disease. "
                "Some fields use defaults when not supplied; interpret the estimate accordingly."
            ),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Cognitive-symptom prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Cognitive-symptom prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PRIVACY-FIRST RESEARCH DATA + SERVER-SIDE AUTHENTICATION
# Required Render environment variables:
# SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, AUTH_SECRET,
# RESEARCHER_PASSWORD, ADMIN_PASSWORD
# Never put the service-role key or access passwords in the HTML.
# ═══════════════════════════════════════════════════════════════════════

ALLOWED_RESEARCH_RECORD_TYPES = {
    "stroke_predictions", "alzheimer_predictions", "cognitive_symptom_assessments",
    "nutrition_tracker", "stress_assessments", "pain_stroke_study"
}
IDENTITY_KEYS = {
    "name", "full_name", "patient_name", "participant_name", "email", "phone",
    "phone_number", "address", "hospital_number", "national_id", "passport_number",
    "date_of_birth", "contact_details", "notes", "comment", "comments", "free_text", "narrative", "custom_notes", "created_at", "submitted_at", "registered_at"
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _auth_secret() -> bytes:
    secret = os.getenv("AUTH_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="Authentication is not configured. Set AUTH_SECRET (at least 32 characters) in the backend environment.")
    return secret.encode("utf-8")


def _make_token(claims: dict, expires_seconds: int = 43200) -> str:
    now = int(time.time())
    payload = {**claims, "iat": now, "exp": now + expires_seconds}
    encoded = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64url(hmac.new(_auth_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return encoded + "." + signature


def _read_token(token: str) -> dict:
    try:
        encoded, supplied_sig = token.split(".", 1)
        expected_sig = _b64url(hmac.new(_auth_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied_sig, expected_sig):
            raise ValueError("bad signature")
        padding = "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded + padding))
        if int(claims.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        if claims.get("role") not in {"participant", "researcher", "admin"}:
            raise ValueError("bad role")
        return claims
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Session is invalid or expired. Please sign in again.")


def _claims_from_request(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    return _read_token(authorization.split(" ", 1)[1].strip())


def _require_researcher(authorization: Optional[str]) -> dict:
    claims = _claims_from_request(authorization)
    if claims.get("role") not in {"researcher", "admin"}:
        raise HTTPException(status_code=403, detail="Researcher or administrator access required.")
    return claims




def _supabase_rest(
    table: str,
    method: str = "POST",
    payload=None,
    query: str = "",
    prefer: str = "return=minimal",
):
    base_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not base_url or not service_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Research storage is not configured. Set SUPABASE_URL "
                "and SUPABASE_SERVICE_ROLE_KEY on the backend."
            ),
        )

    url = f"{base_url}/rest/v1/{table}"
    if query:
        url += f"?{query}"

    body = (
        None
        if payload is None
        else json.dumps(payload, allow_nan=False).encode("utf-8")
    )

   
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": prefer,
    }

# Explicitly target the public schema, where our tables exist.
    if method.upper() in ("GET", "HEAD"):
        headers["Accept-Profile"] = "public"
    else:
        headers["Content-Profile"] = "public"

    # Log request details without exposing credentials or participant data.
    logger.info(
        "Supabase request: method=%s table=%s host=%s key_present=%s",
        method,
        table,
        urllib.parse.urlparse(url).netloc,
        bool(service_key),
    )

    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else None

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]

        logger.error(
            "SUPABASE DIAGNOSTIC | method=%s | url=%s | "
            "status=%s | reason=%s | response_body=%r | "
            "response_headers=%r",
            method,
            url,
            exc.code,
            exc.reason,
            detail or "<empty response body>",
            dict(exc.headers) if exc.headers else {},
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Research database operation failed "
                f"(Supabase HTTP {exc.code}). Check backend logs."
            ),
        ) from exc

    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error(
            "Supabase connection failure: method=%s table=%s error=%r",
            method,
            table,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Research database is temporarily unavailable.",
        ) from exc

    except (ValueError, TypeError) as exc:
        logger.error(
            "Supabase request preparation or response parsing failed: "
            "table=%s error=%r",
            table,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Research request could not be processed.",
        ) from exc


class ResearchParticipantCreate(BaseModel):
    consent: bool
    consent_version: str = "ANHRL-NRF-2026-v1"
    age: Optional[int] = None
    gender: Optional[str] = None
    country: str
    province: Optional[str] = None
    region: Optional[str] = None
    study_site: Optional[str] = None
    geopolitical_zone: Optional[str] = None


class ResearchRecordCreate(BaseModel):
    record_type: str
    payload: dict
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class ResearchOutcomeCreate(BaseModel):
    research_id: str
    outcome_type: str
    outcome_value: str
    assessed_at: Optional[datetime] = None
    label_source: Optional[str] = None
    verified_by_code: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = "researcher"


@app.post("/auth/login")
def auth_login(payload: LoginRequest):
    requested_role = payload.role.lower().strip()
    username = payload.username.strip()[:120]
    password = payload.password
    if requested_role not in {"researcher", "admin"}:
        raise HTTPException(status_code=400, detail="Unsupported account role.")
    env_name = "ADMIN_PASSWORD" if requested_role == "admin" else "RESEARCHER_PASSWORD"
    expected = os.getenv(env_name, "")
    if len(expected) < 12:
        raise HTTPException(status_code=503, detail=f"{env_name} is not configured securely on the backend.")
    if not username or not hmac.compare_digest(password, expected):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    claims = {"sub": username, "role": requested_role}
    token = _make_token(claims, expires_seconds=28800)
    return {"access_token": token, "token_type": "bearer", "role": requested_role, "expires_in": 28800}


@app.get("/auth/verify")
def auth_verify(authorization: Optional[str] = Header(default=None)):
    claims = _claims_from_request(authorization)
    return {"authenticated": True, "role": claims["role"], "subject": claims.get("sub")}


@app.get("/debug/research-table")
def debug_research_table():
    result = _supabase_rest(
        "neurohealth_research_records",
        "GET",
        query="select=id&limit=1",
        prefer="return=representation",
    )
    return {
        "reachable": True,
        "rows_returned": len(result or [])
    }


@app.post("/research/participants")
def create_research_participant(payload: ResearchParticipantCreate):
    """Create a pseudonymous participant ID without collecting direct identifiers."""

    if not payload.consent:
        raise HTTPException(
            status_code=400,
            detail="Participant consent is required before creating a research record."
        )

    country = payload.country.strip()
    if not country or len(country) > 100:
        raise HTTPException(
            status_code=422,
            detail="A valid country is required."
        )

    if payload.age is not None and not 1 <= payload.age <= 120:
        raise HTTPException(
            status_code=422,
            detail="Age must be between 1 and 120, or left blank."
        )

    _auth_secret()

    research_id = "ANH-" + uuid.uuid4().hex.upper()

    # Match the actual neurohealth_research_records table.
    research_row = {
        "research_id": research_id,
        "country": country,
        "african_subregion": (payload.region or "")[:100] or None,
        "nigeria_geopolitical_zone": (
            (payload.geopolitical_zone or "")[:100] or None
        ),
        "state_province": (payload.province or "")[:100] or None,
        "study_site": (payload.study_site or "")[:150] or None,
        "assessment_type": "participant_registration",
        "data_json": {
            "age": payload.age,
            "gender": (payload.gender or "")[:40] or None,
        },
        "consent_confirmed": True,
        "consent_version": (payload.consent_version or "")[:80] or None,
    }

    # Save the record before issuing the participant token.
    _supabase_rest(
        "neurohealth_research_records",
        "POST",
        [research_row],
        prefer="return=minimal"
    )

    token = _make_token(
        {"sub": research_id, "role": "participant"},
        expires_seconds=43200
    )

    return {
        "research_id": research_id,
        "participant_token": token,
        "expires_in": 43200,
        "message": (
            "Research ID created. No direct identity details were requested."
        )
    }


@app.post("/research/records")
def save_research_record(payload: ResearchRecordCreate, authorization: Optional[str] = Header(default=None)):
    """Store only de-identified study variables in the separate research dataset."""
    claims = _claims_from_request(authorization)
    record_type = payload.record_type.strip()
    if record_type not in ALLOWED_RESEARCH_RECORD_TYPES:
        raise HTTPException(status_code=400, detail="This record type is not approved for the research dataset.")
    if claims.get("role") == "participant":
        research_id = claims.get("sub")
    elif claims.get("role") in {"researcher", "admin"}:
        research_id = str(payload.payload.get("research_id", "")).strip()
    else:
        raise HTTPException(status_code=403, detail="This session cannot save research records.")
    if not research_id or len(research_id) > 80:
        raise HTTPException(status_code=400, detail="A valid research ID is required.")
    if len(json.dumps(payload.payload, default=str)) > 100_000:
        raise HTTPException(status_code=413, detail="Research record is too large.")

    normalized_identity_keys = {
        "name", "fullname", "patientname", "participantname", "email", "emailaddress",
        "phone", "phonenumber", "address", "hospitalnumber", "nationalid", "passportnumber",
        "dateofbirth", "contactdetails", "notes", "comment", "comments", "freetext",
        "narrative", "customnotes", "createdat", "submittedat", "registeredat", "userid", "researchid"
    }

    def scrub(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, nested_value in value.items():
                key_text = str(key)[:100]
                normalized = re.sub(r"[^a-z0-9]", "", key_text.lower())
                if normalized in normalized_identity_keys or any(term in normalized for term in ("patientname", "participantname", "emailaddress", "phonenumber")):
                    continue
                cleaned[key_text] = scrub(nested_value)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value[:500]]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return None

    safe_payload = scrub(payload.payload)
    row = {
        "research_id": research_id,
        "record_type": record_type,
        "payload": safe_payload,
        "model_name": (payload.model_name or "")[:120] or None,
        "model_version": (payload.model_version or "")[:80] or None,
    }
    _supabase_rest("research_records", "POST", [row], prefer="return=minimal")
    return {"saved": True, "research_id": research_id, "record_type": record_type}


@app.get("/research/records")
def get_research_records(authorization: Optional[str] = Header(default=None), limit: int = 500):
    _require_researcher(authorization)
    limit = max(1, min(int(limit), 2000))
    query = urlencode({"select": "id,research_id,record_type,payload,model_name,model_version,created_at",
                       "order": "created_at.desc", "limit": str(limit)})
    rows = _supabase_rest("research_records", "GET", query=query, prefer="return=representation") or []
    return {"count": len(rows), "records": rows}


@app.get("/research/participants/summary")
def research_participant_summary(authorization: Optional[str] = Header(default=None)):
    _require_researcher(authorization)
    query = urlencode({"select": "research_id", "limit": "2000"})
    participants = _supabase_rest("research_participants", "GET", query=query, prefer="return=representation") or []
    query_records = urlencode({"select": "record_type", "limit": "5000"})
    records = _supabase_rest("research_records", "GET", query=query_records, prefer="return=representation") or []
    query_outcomes = urlencode({"select": "outcome_type", "limit": "5000"})
    outcomes = _supabase_rest("research_outcomes", "GET", query=query_outcomes, prefer="return=representation") or []
    counts = {}
    for row in records:
        counts[row.get("record_type", "unknown")] = counts.get(row.get("record_type", "unknown"), 0) + 1
    outcome_counts = {}
    for row in outcomes:
        outcome_counts[row.get("outcome_type", "unknown")] = outcome_counts.get(row.get("outcome_type", "unknown"), 0) + 1
    return {"participants_returned": len(participants), "records_returned": len(records), "records_by_type": counts,
            "outcomes_returned": len(outcomes), "outcomes_by_type": outcome_counts,
            "note": "Counts reflect the rows returned by the configured query limits."}


@app.get("/research/map/aggregates")
def research_map_aggregates(layer: str = "stroke", authorization: Optional[str] = Header(default=None)):
    """Return grouped aggregates for the protected NeuroMap, never participant-level rows."""
    _require_researcher(authorization)
    layer = (layer or "stroke").lower().strip()
    allowed_types = {"stroke": {"stroke_predictions"}, "dementia": {"alzheimer_predictions"}}
    if layer not in allowed_types:
        return {"aggregates": [], "min_group_size": 10, "note": "This layer uses configured field-survey summaries; no participant-level data are returned."}
    participant_query = urlencode({"select":"research_id,country,province,region,geopolitical_zone,study_site", "limit":"5000"})
    record_query = urlencode({"select":"research_id,record_type,payload", "record_type":"in.({})".format(",".join(sorted(allowed_types[layer]))), "order":"created_at.desc", "limit":"5000"})
    participants = _supabase_rest("research_participants", "GET", query=participant_query, prefer="return=representation") or []
    records = _supabase_rest("research_records", "GET", query=record_query, prefer="return=representation") or []
    by_id = {str(p.get("research_id")):p for p in participants if p.get("research_id")}
    grouped = {}
    for record in records:
        participant = by_id.get(str(record.get("research_id")))
        if not participant: continue
        country = str(participant.get("country") or "").strip()
        province = str(participant.get("province") or "").strip()
        if country.lower() == "nigeria" and province: area, group_province = province, province
        else: area, group_province = country, None
        if not area: continue
        key = (country, area, record.get("record_type"))
        group = grouped.setdefault(key, {"country":country,"province":group_province,"region":participant.get("region"),"geopolitical_zone":participant.get("geopolitical_zone"),"area":area,"record_type":record.get("record_type"),"n":0,"high_risk_n":0})
        group["n"] += 1
        data = record.get("payload") or {}
        if isinstance(data, dict) and str(data.get("risk_level", "")).upper() == "HIGH": group["high_risk_n"] += 1
    min_group_size = 10
    aggregates = []
    for group in grouped.values():
        if group["n"] < min_group_size: continue
        group["high_risk_pct"] = round(100 * group["high_risk_n"] / group["n"], 1) if group["n"] else 0
        group.pop("high_risk_n", None)
        aggregates.append(group)
    return {"aggregates":aggregates,"min_group_size":min_group_size,"note":"Aggregates only. Groups smaller than 10 records are suppressed. Model outputs are not disease prevalence estimates."}


@app.post("/research/outcomes")
def save_research_outcome(payload: ResearchOutcomeCreate, authorization: Optional[str] = Header(default=None)):
    """Store a structured, verified follow-up label for future model evaluation/retraining."""
    _require_researcher(authorization)
    research_id = payload.research_id.strip()
    outcome_type = payload.outcome_type.strip().lower()
    outcome_value = payload.outcome_value.strip()
    if not research_id.startswith("ANH-") or len(research_id) > 80:
        raise HTTPException(status_code=422, detail="A valid research ID is required.")
    if not outcome_type or len(outcome_type) > 80 or not outcome_value or len(outcome_value) > 100:
        raise HTTPException(status_code=422, detail="Outcome type and categorical outcome value are required.")
    row = {
        "research_id": research_id,
        "outcome_type": outcome_type,
        "outcome_value": outcome_value,
        "assessed_at": payload.assessed_at.isoformat() if payload.assessed_at else None,
        "label_source": (payload.label_source or "")[:100] or None,
        "verified_by_code": (payload.verified_by_code or "")[:80] or None,
    }
    _supabase_rest("research_outcomes", "POST", [row], prefer="return=minimal")
    return {"saved": True, "research_id": research_id, "outcome_type": outcome_type,
            "note": "Use verified study outcomes for model evaluation; do not treat model predictions as ground-truth labels."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
