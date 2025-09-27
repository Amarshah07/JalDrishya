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

# -----------------------
# Config & Styles
# -----------------------
st.set_page_config(page_title="FloatChat - JalDrishya (Local LLM)", page_icon="⚓", layout="wide")

st.markdown("""
<style>
:root {
  --ocean-bg: #CAF0F8;
  --text-primary: #003554;
  --accent: #00B4D8;
}
.stApp { background: linear-gradient(180deg,#e6fbff, #caf0f8); color: var(--text-primary); font-family: "Segoe UI", Roboto, sans-serif; }
.stTitle { font-weight:700 !important; }
.stButton>button { background: linear-gradient(90deg,#00b4d8,#0077b6); color: #fff; border-radius:10px; padding:8px 16px; }
.stMarkdown, .stText { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { background: #90E0EF; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Paths & Knowledge
# -----------------------
DB_PATH = "argo_data.db"

CITY_COORDS = {
    "mumbai": {"lat": 19.07, "lon": 72.87},
    "chennai": {"lat": 13.08, "lon": 80.27},
    "kochi": {"lat": 9.93, "lon": 76.26},
    "visakhapatnam": {"lat": 17.68, "lon": 83.21},
    "kolkata": {"lat": 22.57, "lon": 88.36},
    "goa": {"lat": 15.29, "lon": 74.12}
}

# Short canonical months set (validate user inputs)
MONTHS = {
    "january":"January","february":"February","march":"March","april":"April","may":"May","june":"June",
    "july":"July","august":"August","september":"September","october":"October","november":"November","december":"December"
}

DB_SCHEMA = """
Table Name: argo_profiles
Columns: id (INTEGER), file_name (TEXT), month (TEXT), cycle_number (INTEGER),
latitude (REAL), longitude (REAL), pressure (REAL), temperature (REAL),
salinity (REAL), date (TEXT)
"""

# -----------------------
# Ollama LLM wrapper (robust)
# -----------------------
def call_ollama_json(user_query: str, model_name: str = "llama3.1:8b"):
    """
    Call Ollama model to generate a JSON object with keys:
      - analysis
      - visualization_type
      - sql_query

    Returns dict {"response": parsed_json} or {"error": message, "raw": raw_text}
    """
    system_prompt = (
        "You are a strict SQL generator for an SQLite table named 'argo_profiles'.\n"
        "Return ONLY a valid JSON object (no explanations) with exactly these keys:\n"
        '  "analysis": short one-sentence summary,\n'
        '  "visualization_type": one of ["MAP","PROFILE","BAR_CHART","METRIC","TABLE"],\n'
        '  "sql_query": a valid SQLite query string.\n'
        "Important: Do NOT include any text outside the JSON. Use single SQL statements only.\n"
    )

    user_prompt = f"User Query: \"{user_query}\"\n\nSchema:\n{DB_SCHEMA}\n\nReturn only JSON."

    try:
        # send system + user roles (helps instruct chat models)
        res = ollama.chat(model=model_name, messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}])
    except Exception as e:
        return {"error": f"Ollama call failed: {e}"}

    # try to extract raw text from common response shapes
    raw_text = ""
    if isinstance(res, dict):
        # possible keys
        if "content" in res and res["content"]:
            raw_text = res["content"]
        elif "message" in res:
            msg = res["message"]
            # msg might be dict-like or object-like
            if isinstance(msg, dict):
                raw_text = msg.get("content","") or msg.get("text","") or str(msg)
            else:
                raw_text = getattr(msg,"content", None) or str(msg)
        else:
            raw_text = str(res)
    else:
        raw_text = str(res)

    raw_text = raw_text.strip()
    if not raw_text:
        return {"error": "Ollama returned empty response.", "raw": raw_text}

    # debug: display raw output (small, so safe)
    st.text_area("🛠 LLM raw output (debug)", raw_text, height=150)

    # extract json substring (from first '{' to last '}')
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"error": "Could not find JSON in LLM output.", "raw": raw_text}

    json_string = raw_text[start:end+1]

    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": raw_text}

    # basic validation
    if not all(k in parsed for k in ("analysis","visualization_type","sql_query")):
        return {"error": "LLM JSON missing required keys.", "raw": raw_text, "parsed": parsed}

    return {"response": parsed}

# -----------------------
# Deterministic fallback parser (safe)
# -----------------------
def deterministic_parse(query: str, radius_deg: int = 5):
    """
    If LLM fails, use rule-based parser to generate a reasonable SQL query and analysis.
    Supports:
      - cycle number queries
      - floats near <city>
      - compare average <var> between months
      - profile plot requests
      - max/min metric
    """
    q = query.lower()
    # cycle number
    m = re.search(r"cycle\s*(?:number)?\s*(?:is|=)?\s*(\d{1,6})", q)
    if m:
        cyl = int(m.group(1))
        analysis = f"Show all data for cycle number {cyl}."
        sql = f"SELECT * FROM argo_profiles WHERE cycle_number = {cyl};"
        return {"analysis": analysis, "visualization_type":"TABLE", "sql_query": sql}

    # floats near city
    for city, coords in CITY_COORDS.items():
        if city in q:
            lat = coords['lat']; lon = coords['lon']
            lat1 = lat - radius_deg; lat2 = lat + radius_deg
            lon1 = lon - radius_deg; lon2 = lon + radius_deg
            analysis = f"Show floats within ±{radius_deg}° of {city.title()} (approx)."
            sql = ("SELECT DISTINCT file_name, latitude, longitude, pressure, temperature, salinity, date "
                   f"FROM argo_profiles WHERE latitude BETWEEN {lat1} AND {lat2} "
                   f"AND longitude BETWEEN {lon1} AND {lon2} LIMIT 500;")
            return {"analysis": analysis, "visualization_type":"MAP", "sql_query": sql}

    # compare average variable between months
    # detect variable (salinity/temperature) and months
    var = None
    if "salinity" in q: var = "salinity"
    if "temperature" in q: var = "temperature"
    months_found = re.findall(r"(january|february|march|april|may|june|july|august|september|october|november|december)", q)
    months_found = [MONTHS[m] for m in months_found] if months_found else []
    if var and len(months_found) >= 2 and ("compare" in q or "average" in q):
        months_list = "','".join(months_found[:12])
        analysis = f"Compare average {var} between {', '.join(months_found[:2])}."
        sql = (f"SELECT month, AVG({var}) as avg_{var} FROM argo_profiles "
               f"WHERE month IN ('{months_list}') GROUP BY month;")
        return {"analysis": analysis, "visualization_type":"BAR_CHART", "sql_query": sql}

    # profile request (salinity/temperature profile near X in Month Year)
    if ("profile" in q) or ("profiles" in q):
        var = "salinity" if "salinity" in q else ("temperature" if "temperature" in q else None)
        # month & year
        mm = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", q)
        yy = re.search(r"\b(20\d{2})\b", q)
        # location: region or city
        # if "equator" in q => set bounding box lat -5:5 lon -180:180 (we'll keep lon broad)
        if "equator" in q:
            lat_cond = "latitude BETWEEN -5 AND 5"
            lon_cond = "longitude BETWEEN -180 AND 180"
        else:
            # find city
            found_city = None
            for city, coords in CITY_COORDS.items():
                if city in q:
                    found_city = city; lat = coords['lat']; lon = coords['lon']
                    lat_cond = f"latitude BETWEEN {lat - radius_deg} AND {lat + radius_deg}"
                    lon_cond = f"longitude BETWEEN {lon - radius_deg} AND {lon + radius_deg}"
                    break
            if not found_city:
                # fallback to global
                lat_cond = "1=1"; lon_cond = "1=1"

        where_clauses = [lat_cond, lon_cond]
        if mm:
            month_name = MONTHS[mm.group(1)]
            where_clauses.append(f"month = '{month_name}'")
        if yy:
            where_clauses.append(f"strftime('%Y', date) = '{yy.group(1)}'")

        where_sql = " AND ".join(where_clauses)
        select_cols = "file_name, pressure"
        if var:
            select_cols += f", {var}"
        else:
            select_cols += ", salinity, temperature"

        sql = f"SELECT {select_cols} FROM argo_profiles WHERE {where_sql} ORDER BY file_name, pressure ASC LIMIT 2000;"
        analysis = f"Profile plot request for {var or 'salinity and temperature'}."
        return {"analysis": analysis, "visualization_type":"PROFILE", "sql_query": sql}

    # highest/lowest metric
    if "highest temperature" in q or "max temperature" in q:
        sql = "SELECT MAX(temperature) as max_temperature FROM argo_profiles;"
        return {"analysis": "Find highest recorded temperature.", "visualization_type":"METRIC", "sql_query": sql}
    if "lowest temperature" in q or "min temperature" in q:
        sql = "SELECT MIN(temperature) as min_temperature FROM argo_profiles;"
        return {"analysis": "Find lowest recorded temperature.", "visualization_type":"METRIC", "sql_query": sql}

    # fallback: show table limited
    sql = "SELECT * FROM argo_profiles LIMIT 200;"
    return {"analysis":"Fallback: show sample rows.", "visualization_type":"TABLE", "sql_query": sql}

# -----------------------
# Query executor & visuals
# -----------------------
def execute_sql_query(sql_query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return {"data": df}
    except Exception as e:
        return {"error": f"Database error: {e}", "sql": sql_query}

def plot_map(df: pd.DataFrame):
    if df.empty or not {'latitude','longitude'}.issubset(set(df.columns)):
        st.warning("No lat/lon to plot.")
        return
    center = [float(df['latitude'].mean()), float(df['longitude'].mean())]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    for _, r in df.iterrows():
        popup = f"File: {r.get('file_name','N/A')}<br>Date: {r.get('date','N/A')}<br>Temp: {r.get('temperature','N/A')}<br>Sal: {r.get('salinity','N/A')}"
        folium.CircleMarker([r['latitude'], r['longitude']], radius=4, color="#0069c0", fill=True, fill_color="#00b4d8", popup=folium.Popup(popup, max_width=300)).add_to(m)
    st_folium(m, use_container_width=True, height=500)

def plot_profiles(df: pd.DataFrame):
    if df.empty:
        st.warning("No profile data to plot.")
        return
    # pivot by file_name for multiple profiles
    if 'pressure' in df.columns:
        # create separate plots for temperature and salinity if present
        if 'temperature' in df.columns:
            fig_t = px.line(df, x='temperature', y='pressure', color='file_name', title='Temperature vs Pressure', template='plotly_white')
            fig_t.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_t, use_container_width=True)
        if 'salinity' in df.columns:
            fig_s = px.line(df, x='salinity', y='pressure', color='file_name', title='Salinity vs Pressure', template='plotly_white')
            fig_s.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.warning("Profile data missing 'pressure' column.")

def plot_monthly_bar(df: pd.DataFrame):
    if df.empty or 'month' not in df.columns:
        st.warning("No monthly data.")
        return
    # assume df has month and one metric column
    metric_cols = [c for c in df.columns if c != 'month']
    if not metric_cols:
        st.warning("No metric found to plot.")
        return
    metric = metric_cols[0]
    fig = px.bar(df, x='month', y=metric, title=f"Average {metric} by month", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def display_metric(df: pd.DataFrame):
    if df.shape[1] == 1 and df.shape[0] >= 1:
        val = df.iloc[0,0]
        col = df.columns[0]
        st.metric(label=col.replace('_',' ').title(), value=f"{val:.3f}" if isinstance(val,(float,int)) else str(val))
    else:
        st.write(df)

# -----------------------
# Main UI
# -----------------------
st.title("⚓ FloatChat — JalDrishya (Local Ollama + Fallback)")

st.markdown("Ask natural-language queries about ARGO floats. Examples: *Show me floats near Chennai*, *Show all data for cycle number 205*, *Compare average salinity between January and February*.")

query = st.text_input("Ask your question:", placeholder="e.g., 'Show me floats near Mumbai'")

if query:
    st.info("Processing your query — trying local LLM (llama3.1:8b) first...")
    processed_query = query.strip()

    # try LLM
    llm_result = call_ollama_json(processed_query)

    if "response" in llm_result:
        parsed = llm_result["response"]
        used_llm = True
    else:
        st.warning("LLM failed or returned invalid JSON — using deterministic fallback.")
        parsed = deterministic_parse(processed_query)
        used_llm = False

    # show summary
    st.markdown("**LLM / Parser output**")
    st.json(parsed)

    # execute SQL safely
    sql_to_run = parsed.get("sql_query")
    if not sql_to_run:
        st.error("No SQL generated.")
    else:
        db_res = execute_sql_query(sql_to_run)
        if "error" in db_res:
            st.error(db_res["error"])
            if "sql" in db_res:
                st.code(db_res["sql"])
        else:
            df = db_res["data"]
            st.markdown("---")
            st.success(f"Query returned {len(df)} rows.")
            viz = parsed.get("visualization_type","TABLE")
            if viz == "MAP":
                st.markdown("#### 🗺 Map")
                plot_map(df)
            elif viz == "PROFILE":
                st.markdown("#### 📈 Profile")
                plot_profiles(df)
            elif viz == "BAR_CHART":
                st.markdown("#### 📊 Bar Chart")
                plot_monthly_bar(df)
            elif viz == "METRIC":
                st.markdown("#### 🔢 Metric")
                display_metric(df)
            else:
                st.markdown("#### 📋 Data Table (sample)")
                st.dataframe(df, use_container_width=True)

            # Show the SQL and whether LLM used
            with st.expander("🔎 Query & Diagnostics"):
                st.markdown(f"- Used LLM: **{used_llm}**")
                st.markdown("**Generated SQL:**")
                st.code(sql_to_run)
                if not used_llm:
                    st.markdown("**Note:** this SQL was generated by the fallback parser (deterministic rules).")
