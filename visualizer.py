# visualizer.py
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd

def create_map(df: pd.DataFrame):
    if df.empty or not {'latitude','longitude'}.issubset(df.columns): return None
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    Draw(export=False, position='topleft', draw_options={'polyline': False, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': False, 'circlemarker': False}).add_to(m)
    for _, r in df.iterrows():
        folium.CircleMarker([r['latitude'], r['longitude']], radius=4, color="#0069c0", fill=True, fill_color="#00b4d8", tooltip=f"Float: {r.get('file_name', 'N/A')}<br>Lat: {r['latitude']:.2f}, Lon: {r['longitude']:.2f}").add_to(m)
    return m

def plot_attractive_profile(df: pd.DataFrame):
    if df.empty or 'pressure' not in df.columns: return
    is_monthly_avg = df['file_name'].nunique() == 1 and df['file_name'].iloc[0] == 'Average Profile'
    title = '<b>Average Temperature vs. Depth Profile</b>' if is_monthly_avg else '<b>Temperature vs. Depth Profile by Float ID</b>'
    labels = {'temperature': 'Avg. Temperature (°C)' if is_monthly_avg else 'Temperature (°C)', 'pressure': 'Depth (dbar)', 'file_name': 'Profile ID'}
    fig = px.line(df, x='temperature', y='pressure', color='file_name', title=title, labels=labels, markers=True, color_discrete_sequence=['#00B4D8', '#0077B6', '#90E0EF'])
    fig.update_layout(font_family="Segoe UI", title_font_color="#003554", legend_title_font_color="#003554", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255, 255, 255, 0.6)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(autorange="reversed", gridcolor='rgba(0, 53, 84, 0.2)')
    fig.update_xaxes(gridcolor='rgba(0, 53, 84, 0.2)')
    st.plotly_chart(fig, use_container_width=True)

# (Add other plotting functions like plot_monthly_bar here if needed)