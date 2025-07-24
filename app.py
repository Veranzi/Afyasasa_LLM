# 📦 app.py — FastAPI Integration
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# === Load .env ===
load_dotenv()

# === Load Artifacts ===
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
features = joblib.load("final_features.joblib")
numerical = joblib.load("numerical_columns.joblib")
target_le = joblib.load("target_encoder.joblib")
top5_features = pd.read_json("top5_features.json")

# === Gemini Init ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("models/gemini-1.5-pro-latest")
chat = gemini.start_chat()

# === FastAPI Init ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://afya-sasa.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Schemas ===
class PredictRequest(BaseModel):
    age: float
    menopause_status: str
    cyst_size_cm: float
    cyst_growth_rate: float
    ca125_level: float
    ultrasound_features: str
    symptoms: str

class FollowUpRequest(BaseModel):
    previous_context: str
    question: str

class QueryRequest(BaseModel):
    query: str

# === Prediction Endpoint ===
@app.post("/predict-management")
def predict_ovarian_cyst(request: PredictRequest):
    menopause_map = {"Post-menopausal": 0, "Pre-menopausal": 1}
    ultrasound_map = {
        'Septated cyst': 0,
        'Hemorrhagic cyst': 1,
        'Solid mass': 2,
        'Complex cyst': 3,
        'Simple cyst': 4
    }

    df = pd.DataFrame([{
        "Age": request.age,
        "Menopause Status": menopause_map.get(request.menopause_status, -1),
        "Cyst Size cm": request.cyst_size_cm,
        "Cyst Growth Rate cm/month": request.cyst_growth_rate,
        "CA 125 Level": request.ca125_level,
        "Ultrasound Features": ultrasound_map.get(request.ultrasound_features, -1)
    }])

    symptom_list = [s.strip() for s in request.symptoms.split(",")]
    for col in features:
        if col not in df.columns:
            df[col] = 1 if col in symptom_list else 0

    df = df[features]
    df[numerical] = scaler.transform(df[numerical])

    probs = model.predict_proba(df)[0]
    pred_index = np.argmax(probs)
    pred_class = target_le.inverse_transform([pred_index])[0]
    confidence = float(round(100 * probs[pred_index], 2))
    prob_dict = {target: float(round(p, 4)) for target, p in zip(target_le.classes_, probs)}

    # LLM interpretation
    inventory_df = pd.read_csv("https://docs.google.com/spreadsheets/d/120UNDtWijskCdvZmrCGPTzMHNrk-Yl6-/export?format=csv")
    treatment_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1cdhXfpLZw_3_yuu9wcpI0-50i0o6bnn-/export?format=csv")
    prompt = f"""
    Model Prediction: {pred_class} ({confidence}%)\n
    Probabilities: {prob_dict}\n
    Inventory:\n{inventory_df.to_string(index=False)}\n
    NHIF Costs:\n{treatment_df.to_string(index=False)}\n
    Based on this, generate a clinical follow-up recommendation.
    """
    followup = chat.send_message(prompt).text.strip()

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "interpretation": followup
    }

# === Follow-Up Endpoint ===
@app.post("/follow-up")
def follow_up(request: FollowUpRequest):
    prompt = f"Previous:\n{request.previous_context}\nFollow-Up Question:\n{request.question}"
    return {"followup_response": chat.send_message(prompt).text.strip()}

# === General Ask Endpoint ===
@app.post("/ask")
def general_ovarian_assistant(request: QueryRequest):
    query = request.query
    inventory_df = pd.read_csv("https://docs.google.com/spreadsheets/d/120UNDtWijskCdvZmrCGPTzMHNrk-Yl6-/export?format=csv")
    treatment_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1cdhXfpLZw_3_yuu9wcpI0-50i0o6bnn-/export?format=csv")

    inventory_match = inventory_df[inventory_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    if not inventory_match.empty:
        return {"source": "Inventory Data", "answer": inventory_match.to_string(index=False)}

    treatment_match = treatment_df[treatment_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    if not treatment_match.empty:
        return {"source": "NHIF Treatment Cost", "answer": treatment_match.to_string(index=False)}

    prompt = f"You are a clinical assistant for ovarian cysts. If query is unrelated, say so.\n\nQ: {query}"
    response = chat.send_message(prompt)
    return {"source": "Gemini Reasoning", "answer": response.text.strip()}

# === Root Check ===
@app.get("/")
def root():
    return {"message": "AfyaSasa Prediction API is running!"}
