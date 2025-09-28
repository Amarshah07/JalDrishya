# app.py
import streamlit as st
import pandas as pd
from db_utils import execute_sql_query
from ai_handler import deterministic_parse, call_ollama_json, call_gemini_api, get_fallback_response
from visualizer import create_map, plot_attractive_profile
from streamlit_folium import st_folium

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="FloatChat - JalDrishya", page_icon="⚓", layout="wide")
    load_css("style.css")

    st.title("FloatChat: Intelligent ARGO Data Explorer")
    st.markdown("<p style='text-align: center;'>Ask a question, and the AI will analyze it, choose the best visualization, and show you the answer.</p>", unsafe_allow_html=True)

    if "map_df" not in st.session_state: st.session_state.map_df = pd.DataFrame()

    with st.expander("💡 Example Questions to Try"):
        st.markdown("- **Map:** `Show me floats near Chennai`\n- **Monthly Profile:** `Plot the temperature vs depth graph of January month`\n- **Float Profile:** `Plot the profile for float 2902596`")
    
    query = st.text_input("Ask your question:", placeholder="e.g., 'Show floats near Mumbai'")

    if query:
        st.session_state.map_df = pd.DataFrame()
        
        parsed_by_rules = deterministic_parse(query.strip().lower())
        
        if parsed_by_rules:
            parsed, used_method = parsed_by_rules, "Rule-Based Parser"
        else:
            st.info("🧠 Complex query detected. Trying AI cascade...")
            llama_result = call_ollama_json(query)
            llama_succeeded = False
            
            if "response" in llama_result:
                sql_to_validate = llama_result["response"]["sql_query"]
                validation_res = execute_sql_query(sql_to_validate)
                if "data" in validation_res and not validation_res["data"].empty:
                    parsed, used_method = llama_result["response"], "Local LLM (Ollama)"
                    raw_output, llama_succeeded = llama_result.get("raw", ""), True
            
            if not llama_succeeded:
                st.info("Local AI failed. Escalating to Gemini... 🧠✨")
                gemini_result = call_gemini_api(query)
                raw_output = gemini_result.get("raw", "")
                if "response" in gemini_result:
                    parsed, used_method = gemini_result["response"], "Cloud AI (Gemini)"
                else:
                    st.warning("Advanced AI failed. Using a simple fallback.")
                    parsed, used_method = get_fallback_response(), "Fallback"
        
        st.markdown("**🤖 AI/Parser Output**"); st.json(parsed)
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
                    if viz_type == "MAP": st.session_state.map_df = df
                    elif viz_type == "PROFILE":
                        st.markdown("#### 📈 Profile")
                        plot_attractive_profile(df)
                    else:
                        st.markdown("#### 📋 Data Table")
                        st.dataframe(df)

    if not st.session_state.map_df.empty:
        st.markdown("#### 🗺️ Interactive Map")
        st.info("Use the drawing tools on the left to select a specific area.")
        map_obj = create_map(st.session_state.map_df)
        map_output = st_folium(map_obj, use_container_width=True, height=500, key="map_display")
        
        if map_output and map_output.get("all_drawings"):
            shape = map_output["all_drawings"][0]
            coords = shape["geometry"]["coordinates"][0]
            min_lon, max_lon = min(c[0] for c in coords), max(c[0] for c in coords)
            min_lat, max_lat = min(c[1] for c in coords), max(c[1] for c in coords)
            filtered_df = st.session_state.map_df[(st.session_state.map_df['latitude'].between(min_lat, max_lat)) & (st.session_state.map_df['longitude'].between(min_lon, max_lon))]
            st.markdown("---"); st.markdown(f"#### 📍 {len(filtered_df)} Floats in Selected Area"); st.dataframe(filtered_df)

if __name__ == "__main__":
    main()