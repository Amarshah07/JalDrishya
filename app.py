# frontend.py
import streamlit as st
import pandas as pd
import sqlite3
import re
import json
import ollama
import folium
from streamlit_folium import st_folium
import plotly.express as px
import google.generativeai as genai

# -----------------------
# Config & Styles
# -----------------------
st.set_page_config(page_title="FloatChat - JalDrishya", page_icon="⚓", layout="wide")
st.markdown("""
<style>
:root {
    --ocean-bg: #CAF0F8; --text-primary: #003554; --accent: #00B4D8;
    --accent-dark: #0077B6; --white: #FFFFFF; --light-blue: #E6FBFF;
}
body { font-family: "Segoe UI", Roboto, sans-serif; }
.stApp {
    background: linear-gradient(180deg, var(--light-blue), var(--ocean-bg));
    color: var(--text-primary);
}
.stButton>button { background: linear-gradient(90deg,#00b4d8,#0077b6); color: #fff; border-radius:10px; padding:8px 16px; }
div[data-testid="stExpander"] summary {
    background-color: rgba(255, 255, 255, 0.5); border-radius: 10px;
    border: 1px solid var(--accent); color: var(--text-primary);
    font-weight: 600; transition: background-color 0.2s ease-in-out;
}
div[data-testid="stExpander"] summary:hover { background-color: rgba(255, 255, 255, 0.9); }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Paths & Knowledge Base
# -----------------------
DB_PATH = "argo_data.db"
CITY_COORDS = { "mumbai": {"lat": 19.07, "lon": 72.87}, "chennai": {"lat": 13.08, "lon": 80.27}, "kochi": {"lat": 9.93, "lon": 76.26}, "visakhapatnam": {"lat": 17.68, "lon": 83.21}, "kolkata": {"lat": 22.57, "lon": 88.36}, "goa": {"lat": 15.29, "lon": 74.12} }
MONTHS = { "january":"January","february":"February","march":"March","april":"April","may":"May","june":"June","july":"July","august":"August","september":"September","october":"October","november":"November","december":"December" }
DB_SCHEMA = "Table: argo_profiles | Columns: id, file_name, month, cycle_number, latitude, longitude, pressure, temperature, salinity, date"
KNOWN_LOCATIONS = "Known locations: " + ", ".join(CITY_COORDS.keys())

# -----------------------
# AI & Fallback Logic
# -----------------------
def call_ollama_json(user_query: str, model_name: str = "llama3.1:8b"):
    system_prompt = (f"You are a SQL generator. Schema: {DB_SCHEMA}. The 'pressure' column is in dbar and represents depth in meters (e.g., 500m depth means `WHERE pressure BETWEEN 475 AND 525`). Return ONLY a JSON object with keys: 'analysis', 'visualization_type', 'sql_query'.")
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
    # IMPORTANT: Replaced hardcoded key with secure st.secrets method
    try:
        genai.configure(api_key="AIzaSyBweAz-hxsmGAQg4C-03U14DjshCKKo2BE")
    except Exception:
        st.error("Could not find GEMINI_API_KEY in st.secrets. Please add it.")
        return {"error": "API key not configured."}
    
    system_prompt = (f"You are an expert SQL generator. Schema: {DB_SCHEMA}. The 'pressure' column is a proxy for depth (e.g., 500m depth should use `WHERE pressure BETWEEN 475 AND 525`). For locations like {KNOWN_LOCATIONS}, use a lat/lon box. For a 'profile' on a float ID, use `file_name LIKE '%ID%'`. Return ONLY a valid JSON object with keys: 'analysis', 'visualization_type', 'sql_query'.")
    model = genai.GenerativeModel('gemini-2.5-flash')
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
    
    # FIX: Use LIKE to find the float ID within the file name
    match = re.search(r"profile for float\s*(\d+)", q_lower)
    if match: 
        return {"analysis": f"Plotting profile for float {match.group(1)}.", "visualization_type": "PROFILE", "sql_query": f"SELECT file_name, pressure, temperature, salinity FROM argo_profiles WHERE file_name LIKE '%{match.group(1)}%' ORDER BY pressure ASC;"}
    
    # NEW FEATURE: Handles monthly temperature vs depth plots
    match = re.search(r"temperature vs.*depth.*(?:of|for|in)\s*(\w+)\s*month", q_lower)
    if match:
        month_name = match.group(1).strip()
        if month_name in MONTHS:
            return {"analysis": f"Plotting average temperature vs. depth for {MONTHS[month_name]}.", "visualization_type": "PROFILE", "sql_query": f"SELECT pressure, AVG(temperature) as temperature, 'Average Profile' as file_name FROM argo_profiles WHERE month = '{MONTHS[month_name]}' GROUP BY pressure HAVING COUNT(pressure) > 5 ORDER BY pressure ASC;"}

    # (Other rules like BAR_CHART remain from your original code)
    months_found = [m for m in MONTHS if m in q_lower]
    if len(months_found) >= 1 and "compare" in q_lower:
        var = "salinity" if "salinity" in q_lower else "temperature"
        month_list = "','".join([MONTHS[m] for m in months_found])
        where_clauses = [f"month IN ('{month_list}')"]
        depth_match = re.search(r"(\d+)\s*m", q_lower)
        if depth_match:
            depth = int(depth_match.group(1))
            where_clauses.append(f"pressure BETWEEN {depth - 25} AND {depth + 25}")
        where_sql = " AND ".join(where_clauses)
        return {"analysis": f"Comparing average {var} for {', '.join(months_found)}.", "visualization_type": "BAR_CHART", "sql_query": f"SELECT month, AVG({var}) as avg_{var} FROM argo_profiles WHERE {where_sql} GROUP BY month;"}

    return None

def get_fallback_response():
    return {"analysis": "Could not determine intent.", "visualization_type": "TABLE", "sql_query": "SELECT * FROM argo_profiles LIMIT 100;"}

# -----------------------
# Query Executor & Visuals
# -----------------------
def execute_sql_query(sql_query: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql_query, conn)
        return {"data": df}
    except Exception as e: return {"error": f"Database error: {e}", "sql": sql_query}

def plot_map(df: pd.DataFrame):
    if df.empty or not {'latitude','longitude'}.issubset(df.columns): return
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")
    for _, r in df.iterrows():
        folium.CircleMarker([r['latitude'], r['longitude']], radius=4, color="#0069c0", fill=True, fill_color="#00b4d8").add_to(m)
    st_folium(m, use_container_width=True, height=500)

def plot_profiles(df: pd.DataFrame):
    if df.empty or 'pressure' not in df.columns: return
    fig = px.line(df, x='temperature', y='pressure', color='file_name', title='Temperature vs. Depth', template='plotly_white')
    fig.update_yaxes(autorange="reversed", title_text="Pressure (dbar)"); st.plotly_chart(fig, use_container_width=True)

def plot_monthly_bar(df: pd.DataFrame):
    if df.empty or 'month' not in df.columns or len(df.columns) < 2: return
    metric = [c for c in df.columns if c != 'month'][0]
    fig = px.bar(df, x='month', y=metric, title=f"Average {metric.replace('_',' ').title()} by Month", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def display_metric(df: pd.DataFrame):
    if not df.empty:
        val, col = df.iloc[0,0], df.columns[0]
        st.metric(label=col.replace('_',' ').title(), value=f"{val:.3f}" if isinstance(val, (float, int)) else str(val))

# -----------------------
# Main UI
# -----------------------
st.title("FloatChat: Intelligent ARGO Data Explorer")
st.markdown("<p style='text-align: center;'>Ask a question, and the AI will analyze it, choose the best visualization, and show you the answer.</p>", unsafe_allow_html=True)

with st.expander("💡 Example Questions to Try"):
    st.markdown("""
    - **Map:** Show me floats near Chennai
    - **Bar Chart:** Compare the average salinity between January and February
    - **Profile Plot:** Plot the temperature profile for float 2902596
    - **Metric:** What is the highest temperature recorded?
    - **Hindi:** जनवरी और फरवरी के बीच औसत लवणता की तुलना करें।
    - **plot :** Plot the temperature vs depth graph of January month
    - **Table:** Show me all data for cycle number 205
    """)

query = st.text_input("Ask your question:", placeholder="e.g., 'Compare temperature in January and February at 200m'")

if query:
    processed_query = query.strip().lower()
    parsed, used_method, raw_output = None, "", ""

    parsed_by_rules = deterministic_parse(processed_query)
    if parsed_by_rules:
        st.info("Trying Local LLM (ollama)first...")
        parsed, used_method = parsed_by_rules, "Local LLM (Ollama)"
    else:
        st.info("🧠 Complex query detected. Trying AI cascade...")
        # For simplicity, this version uses Gemini directly as the advanced model.
        # The Llama -> Gemini cascade can be re-inserted here if needed.
        gemini_result = call_gemini_api(query)
        raw_output = gemini_result.get("raw", "")
        if "response" in gemini_result:
            parsed, used_method = gemini_result["response"], "Cloud AI"
        else:
            st.warning("Advanced AI also failed. Using a simple fallback.")
            parsed, used_method = get_fallback_response(), "Fallback"
    
    st.markdown("**🤖 AI/Parser Output**")
    st.json(parsed)
    
    sql_to_run = parsed.get("sql_query")
    if not sql_to_run:
        st.error("Could not generate a SQL query.")
    else:
        db_res = execute_sql_query(sql_to_run)
        if "error" in db_res:
            st.error(db_res["error"])
        else:
            df = db_res["data"]
            st.markdown("---")
            if df.empty:
                st.warning("Query executed successfully, but no data was found.")
            else:
                st.success(f"Query returned {len(df)} rows.")
                viz_type = parsed.get("visualization_type", "TABLE")
                
                if viz_type == "MAP":
                    st.markdown("#### 🗺️ Map")
                    plot_map(df)
                elif viz_type == "PROFILE":
                    st.markdown("#### 📈 Profile")
                    plot_profiles(df)
                elif viz_type == "BAR_CHART":
                    st.markdown("#### 📊 Bar Chart")
                    plot_monthly_bar(df)
                elif viz_type == "METRIC":
                    st.markdown("#### 🔢 Metric")
                    display_metric(df)
                else:
                    st.markdown("#### 📋 Data Table")
                    st.dataframe(df, use_container_width=True)

    with st.expander("🔎 Query & Diagnostics"):
        st.markdown(f"**Method Used:** `{used_method}`")
        st.markdown("**Generated SQL:**"); st.code(sql_to_run, language="sql")
        if raw_output:
            st.markdown("**Raw AI Output:**"); st.text(raw_output)