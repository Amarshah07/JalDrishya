
import streamlit as st
import sqlite3
import pandas as pd
from config import DB_PATH

@st.cache_data
def execute_sql_query(sql_query: str):
    """Connects to the DB, runs the query, and returns a DataFrame or an error."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql_query, conn)
        return {"data": df}
    except Exception as e:
        return {"error": f"Database error: {e}", "sql": sql_query}