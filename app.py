import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import re
import json
import folium
from streamlit_folium import st_folium
import plotly.express as px
import time

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="FloatChat - Intelligent ARGO Explorer",
    page_icon="🌊",
    layout="wide"
)

# Your Gemini API Key
GEMINI_API_KEY = "AIzaSyCGcpzp4miHfLELmIOh2lewLKjRUkScbKY"

# Path to your database
DB_PATH = "argo_data.db"

# --- Location "Knowledge Base" ---
CITY_COORDS = {
    "mumbai": {"lat": 19.07, "lon": 72.87}, "chennai": {"lat": 13.08, "lon": 80.27},
    "kochi": {"lat": 9.93, "lon": 76.26}, "visakhapatnam": {"lat": 17.68, "lon": 83.21},
    "kolkata": {"lat": 22.57, "lon": 88.36}, "goa": {"lat": 15.29, "lon": 74.12}
}

# --- Database Schema Definition ---
DB_SCHEMA = """
Table Name: argo_profiles
Columns:
  - Column: id (INTEGER), file_name (TEXT), month (TEXT), cycle_number (INTEGER),
    latitude (REAL), longitude (REAL), pressure (REAL), temperature (REAL),
    salinity (REAL), date (TEXT)
"""

# Configure the Gemini API client
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error configuring the Google Gemini API: {e}")

# --- 2. REFACTORED SINGLE-CALL LLM LOGIC ---

def get_response_from_gemini(user_query: str):
    """
    A single, efficient LLM call to get both the analysis and the SQL query.
    This reduces API usage by 50% to avoid quota errors.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    You are an expert data analyst and SQLite engineer. Your job is to understand a user's request about ARGO ocean data and respond with a single JSON object containing three keys:
    1. "analysis": A brief, one-sentence summary of what the user wants to do.
    2. "visualization_type": One of ["MAP", "PROFILE", "BAR_CHART", "METRIC", "TABLE"].
    3. "sql_query": The single, valid SQLite query needed to get the data for the analysis.

    RULES:
    - The visualization_type must be "MAP" for locations, "BAR_CHART" for monthly comparisons, "METRIC" for single values (max, min, count), "PROFILE" for depth plots, and "TABLE" for everything else.
    - The `date` column is 'YYYY-MM-DD'.
    - The `month` column is TEXT (e.g., 'January').
    - The unique float ID is in `file_name` (e.g., '...D2902596...'). Query it using `file_name LIKE '%2902596%'`.
    - For queries 'near' a lat/lon, create a bounding box of +/- 5 degrees.
    - For monthly comparisons, the SQL query must calculate the average of the requested metric and GROUP BY month.

    Database Schema: {DB_SCHEMA}

    User Query: "{user_query}"
    JSON Response:
    """
    try:
        response = model.generate_content(prompt)
        json_string = re.sub(r"```json\n|```", "", response.text).strip()
        # FIX: Clean the generated SQL within the parsed JSON
        parsed_json = json.loads(json_string)
        if 'sql_query' in parsed_json and parsed_json['sql_query'].lower().startswith('sqlite'):
            parsed_json['sql_query'] = parsed_json['sql_query'][6:].lstrip()
        return {"response": parsed_json}
    except Exception as e:
        # Check for rate limit error specifically
        if "429" in str(e):
             return {"error": "API rate limit exceeded. Please wait a minute and try again."}
        return {"error": f"LLM Analyst Error: Could not parse the analysis. Details: {e}"}


def execute_sql_query(sql_query: str):
    """Executes the final SQL query on the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return {"data": df}
    except Exception as e:
        return {"error": f"Database error on SQL: '{sql_query}'. Details: {e}"}

# --- 3. DATA VISUALIZATION FUNCTIONS (Unchanged) ---

def plot_map(df: pd.DataFrame):
    """Generates and displays a map of float locations."""
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        return
    map_center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")
    for _, row in df.iterrows():
        popup_html = f"<b>Float ID:</b> {row.get('id', 'N/A')}<br><b>Date:</b> {row.get('date', 'N/A')}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5, color="#0069c0", fill=True, fill_color="#0069c0",
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
    st_folium(m, use_container_width=True, height=500)

def plot_profiles(df: pd.DataFrame):
    """Generates and displays temperature and salinity profile plots."""
    if df.empty: return
    col1, col2 = st.columns(2)
    with col1:
        if 'temperature' in df.columns and 'pressure' in df.columns:
            fig_temp = px.line(df, x='temperature', y='pressure', color='file_name' if 'file_name' in df.columns else None,
                               title='Temperature vs. Depth', template="plotly_white")
            fig_temp.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Temperature (°C)", yaxis_title="Pressure (dbar)")
            st.plotly_chart(fig_temp, use_container_width=True)
    with col2:
        if 'salinity' in df.columns and 'pressure' in df.columns:
            fig_sal = px.line(df, x='salinity', y='pressure', color='file_name' if 'file_name' in df.columns else None,
                              title='Salinity vs. Depth', template="plotly_white")
            fig_sal.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Salinity (PSU)", yaxis_title="Pressure (dbar)")
            st.plotly_chart(fig_sal, use_container_width=True)

def plot_monthly_comparison(df: pd.DataFrame):
    """Generates a bar chart for comparing metrics between months."""
    if df.empty or df.shape[1] < 2 or 'month' not in df.columns:
        return
    st.markdown("#### Monthly Comparison")
    metric_column = [col for col in df.columns if col != 'month'][0]
    fig = px.bar(df, x='month', y=metric_column,
                 title=f"Comparison of Average {metric_column.replace('_', ' ').title()} by Month",
                 template="plotly_white", text_auto='.2f')
    fig.update_traces(marker_color='#0069c0')
    st.plotly_chart(fig, use_container_width=True)
    
def display_metric(df: pd.DataFrame):
    """Displays a single-value result in a prominent metric card."""
    if df.shape == (1, 1):
        single_value = df.iloc[0, 0]
        column_name = df.columns[0]
        st.metric(label=f"Result for: {column_name.replace('_', ' ').title()}", 
                  value=f"{single_value:.2f}" if isinstance(single_value, float) else single_value)

# --- 4. MAIN APPLICATION INTERFACE ---

st.title("🌊 FloatChat: Intelligent ARGO Data Explorer")
st.markdown("<p style='text-align: center;'>Ask a question, and the AI will analyze it, choose the best visualization, and show you the answer.</p>", unsafe_allow_html=True)

with st.expander("💡 Example Questions to Try"):
    st.markdown("""
    - **Map:** `Show me floats near Chennai`
    - **Bar Chart:** `Compare the average salinity between January and February`
    - **Profile Plot:** `Plot the temperature profile for float 2902596`
    - **Metric:** `What is the highest temperature recorded?`
    - **Table:** `Show me all data for cycle number 205`
    """)

# Helper function for location preprocessing
def preprocess_query_for_locations(query: str, radius: int = 5):
    """
    Scans for city names and replaces them with a coordinate-based instruction
    for the AI Analyst, telling it to use a wide search radius.
    """
    for city, coords in CITY_COORDS.items():
        if city in query.lower():
            lat, lon = coords['lat'], coords['lon']
            instruction = (
                f"around the coordinates {lat}, {lon} using a wide search area of +/- {radius} degrees."
            )
            return query.lower().replace(f"near {city}", instruction)
    return query


query = st.text_input("Ask your question:", placeholder="e.g., 'Compare temperature for January and March'", label_visibility="collapsed")

if query:
    with st.spinner("👩‍💻 AI is analyzing your request and generating the response..."):
        processed_query = preprocess_query_for_locations(query)
        
        # --- Single, efficient API call ---
        gemini_result = get_response_from_gemini(processed_query)

    if "error" in gemini_result:
        st.error(gemini_result["error"])
    else:
        response_data = gemini_result['response']
        analysis = response_data.get('analysis', 'Analysis not provided.')
        viz_type = response_data.get('visualization_type', 'TABLE')
        generated_sql = response_data.get('sql_query')

        st.info(f"🤖 **AI Analyst:** {analysis}")

        if not generated_sql:
            st.error("AI did not return a SQL query. Please try rephrasing your question.")
        else:
            db_result = execute_sql_query(generated_sql)

            if "error" in db_result:
                st.error(db_result["error"])
            else:
                df_results = db_result["data"]
                st.markdown("---")

                if df_results.empty:
                    st.warning("Query successful, but no data was found for your specific criteria.")
                else:
                    # --- Intelligent Visualization Routing ---
                    if viz_type == "METRIC":
                        display_metric(df_results)
                    elif viz_type == "BAR_CHART":
                        plot_monthly_comparison(df_results)
                    elif viz_type == "MAP":
                        st.markdown("#### 🗺️ Map View")
                        plot_map(df_results)
                    elif viz_type == "PROFILE":
                        st.markdown("#### 📈 Profile Plots")
                        plot_profiles(df_results)
                    else: # Default to table view
                        st.markdown("#### 📋 Data Table")
                        st.dataframe(df_results, use_container_width=True)

            with st.expander("🤖 View AI's Full Thought Process"):
                st.json(response_data)

