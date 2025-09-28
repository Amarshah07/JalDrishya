# ai_handler.py
import streamlit as st
import re
import json
import ollama
import google.generativeai as genai
from config import DB_SCHEMA, KNOWN_LOCATIONS, CITY_COORDS, MONTHS

def call_ollama_json(user_query: str, model_name: str = "llama3.1:8b"):
    system_prompt = (f"You are a SQL generator. Schema: {DB_SCHEMA}. The 'pressure' column is in dbar and represents depth (e.g., 500m is `WHERE pressure BETWEEN 475 AND 525`). Return ONLY a JSON object with keys: 'analysis', 'visualization_type', 'sql_query'.")
    user_prompt = f"User Query: \"{user_query}\"\n\nJSON Response:"
    try:
        response = ollama.chat(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        raw_text = response['message']['content'].strip()
    except Exception as e: return {"error": f"Ollama API call failed: {e}"}
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match: return {"error": "No JSON in Llama response.", "raw": raw_text}
    try:
        parsed_json = json.loads(json_match.group(0))
        if not all(k in parsed_json for k in ("analysis", "visualization_type", "sql_query")): return {"error": "Llama JSON missing keys.", "raw": raw_text}
        return {"response": parsed_json, "raw": raw_text}
    except json.JSONDecodeError: return {"error": f"Failed to parse Llama JSON.", "raw": raw_text}

def call_gemini_api(user_query: str):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("Could not find GEMINI_API_KEY in st.secrets. Please add it.")
        return {"error": "API key not configured."}
    
    system_prompt = (f"You are an expert SQL generator. Schema: {DB_SCHEMA}. When filtering by month, you MUST use the full month name (e.g., 'January'). The 'pressure' column is a proxy for depth (e.g., 500m depth uses `WHERE pressure BETWEEN 475 AND 525`). For a 'profile' on a float ID, use `file_name LIKE '%ID%'`. Return ONLY a valid JSON object with keys: 'analysis', 'visualization_type', 'sql_query'.")
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(system_prompt + "\n\nUser Query: " + user_query)
        raw_text = response.text.strip()
    except Exception as e: return {"error": f"Gemini API call failed: {e}"}
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match: return {"error": "No JSON in Gemini response.", "raw": raw_text}
    try:
        parsed_json = json.loads(json_match.group(0))
        if not all(k in parsed_json for k in ("analysis", "visualization_type", "sql_query")): return {"error": "Gemini JSON missing keys.", "raw": raw_text}
        return {"response": parsed_json, "raw": raw_text}
    except json.JSONDecodeError: return {"error": f"Failed to parse Gemini JSON.", "raw": raw_text}

def deterministic_parse(query: str, radius_deg: int = 5):
    q_lower = query.lower()
    for city, coords in CITY_COORDS.items():
        if city in q_lower: return {"analysis": f"Showing floats near {city.title()}.", "visualization_type": "MAP", "sql_query": f"SELECT * FROM argo_profiles WHERE latitude BETWEEN {coords['lat']-radius_deg} AND {coords['lat']+radius_deg} AND longitude BETWEEN {coords['lon']-radius_deg} AND {coords['lon']+radius_deg} LIMIT 500;"}
    
    match = re.search(r"profile for float\s*(\d+)", q_lower)
    if match: return {"analysis": f"Plotting profile for float {match.group(1)}.", "visualization_type": "PROFILE", "sql_query": f"SELECT file_name, pressure, temperature, salinity FROM argo_profiles WHERE file_name LIKE '%{match.group(1)}%' ORDER BY pressure ASC;"}
    
    match = re.search(r"temperature vs.*depth.*(?:of|for|in)\s*(\w+)\s*month", q_lower)
    if match:
        month_name = match.group(1).strip()
        if month_name in MONTHS: return {"analysis": f"Plotting average temperature vs. depth for {MONTHS[month_name]}.", "visualization_type": "PROFILE", "sql_query": f"SELECT pressure, AVG(temperature) as temperature, 'Average Profile' as file_name FROM argo_profiles WHERE month = '{MONTHS[month_name]}' GROUP BY pressure HAVING COUNT(pressure) > 5 ORDER BY pressure ASC;"}

    return None

def get_fallback_response():
    return {"analysis": "Could not determine intent.", "visualization_type": "TABLE", "sql_query": "SELECT * FROM argo_profiles LIMIT 100;"}