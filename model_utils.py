import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # This will search for .env in your project and parent folders
print(os.getenv("GEMINI_API_KEY"))
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

genai.configure(api_key=api_key)

def query_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt).text.strip()
    return response