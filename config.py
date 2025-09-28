# config.py

DB_PATH = "data/argo_data.db"

CITY_COORDS = {
    "mumbai": {"lat": 19.07, "lon": 72.87}, "chennai": {"lat": 13.08, "lon": 80.27},
    "kochi": {"lat": 9.93, "lon": 76.26}, "visakhapatnam": {"lat": 17.68, "lon": 83.21},
    "kolkata": {"lat": 22.57, "lon": 88.36}, "goa": {"lat": 15.29, "lon": 74.12}
}

MONTHS = {
    "january":"January","february":"February","march":"March","april":"April","may":"May","june":"June",
    "july":"July","august":"August","september":"September","october":"October","november":"November","december":"December"
}

DB_SCHEMA = "Table: argo_profiles | Columns: id, file_name, month, cycle_number, latitude, longitude, pressure, temperature, salinity, date"
KNOWN_LOCATIONS = "Known locations: " + ", ".join(CITY_COORDS.keys())