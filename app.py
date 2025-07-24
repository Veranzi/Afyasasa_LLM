from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from model_utils import query_gemini
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev, restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ovarian_cyst_knowledge_assistant(request: QueryRequest):
    query = request.query

    inventory_url = "https://docs.google.com/spreadsheets/d/120UNDtWijskCdvZmrCGPTzMHNrk-Yl6-/export?format=csv"
    treatment_url = "https://docs.google.com/spreadsheets/d/1cdhXfpLZw_3_yuu9wcpI0-50i0o6bnn-/export?format=csv"

    inventory_df = pd.read_csv(inventory_url)
    treatment_df = pd.read_csv(treatment_url)

    inventory_match = inventory_df[inventory_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    if not inventory_match.empty:
        return {
            "source": "Inventory Data",
            "answer": f"Found the following item(s) in inventory:\n\n{inventory_match.to_string(index=False)}"
        }

    treatment_match = treatment_df[treatment_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    if not treatment_match.empty:
        return {
            "source": "NHIF Treatment Cost",
            "answer": f"Found relevant cost or service data:\n\n{treatment_match.to_string(index=False)}"
        }

    # Use Gemini for fallback
    prompt = (
        "You are a clinical assistant who ONLY answers questions about ovarian cysts and greetings.\n\n"
        "If the question is unrelated, respond: 'Sorry, I only assist with ovarian cyst-related queries.'\n\n"
        f"Answer this question:\n{query}"
    )
    response = query_gemini(prompt)

    return {
        "source": "AfyaSasa_Bot Reasoning",
        "answer": response
    }
