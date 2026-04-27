import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json
import os
from datetime import datetime

# --- 1. Path & Configuration Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model is in the same folder as this script
ST_MODEL_PATH = os.path.join(BASE_DIR, 'electricity_RF_prediction_model.pkl')

# Absolute path for the CSV on your Desktop
ST_CSV_PATH = r"C:\Users\smroo\OneDrive\Desktop\samrood_projects\weather\dataset\python machine learning model xgboost - electricity demand dataset.csv"

API_KEY = "8H37ZY74J5X2EFZLNJ9LUWKKK"
LOCATION = "11.1457,75.9643" # Karipur, Kerala

st.set_page_config(page_title="Smart Grid Forecast", layout="wide", page_icon="⚡")

# --- 2. Data Fetching & Processing Functions ---

@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}/tomorrow?unitGroup=metric&contentType=json&key={API_KEY}"
    try:
        with urllib.request.urlopen(url) as response:
            raw_data = json.loads(response.read().decode())
        
        forecast_list = []
        for hr in raw_data['days'][0]['hours']:
            forecast_list.append({
                'Timestamp': raw_data['days'][0]['datetime'] + ' ' + hr['datetime'],
                'Temperature': hr['temp'],
                'Humidity': hr['humidity']
            })
        return pd.DataFrame(forecast_list)
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None

def prepare_features(live_df, historical_df):
    live_df['Timestamp'] = pd.to_datetime(live_df['Timestamp'])
    live_df.set_index('Timestamp', inplace=True)
    
    # Time-based engineering
    live_df['hour'] = live_df.index.hour
    live_df['dayofweek'] = live_df.index.dayofweek
    live_df['month'] = live_df.index.month
    live_df['year'] = live_df.index.year
    live_df['dayofyear'] = live_df.index.dayofyear
    live_df['quarter'] = live_df.index.quarter
    live_df['week_of_year'] = live_df.index.isocalendar().week.astype(int)
    live_df['is_weekend'] = live_df.index.dayofweek.isin([5, 6]).astype(int)

    # Lag Logic (Calculated from the end of your CSV)
    last_known_demand = historical_df['Demand'].tail(168).values
    live_df['demand_lag_24hr'] = last_known_demand[-24]
    live_df['demand_lag_168hr'] = last_known_demand[0]
    live_df['demand_rolling_mean_24h'] = np.mean(last_known_demand[-24:])
    live_df['demand_rolling_std_24h'] = np.std(last_known_demand[-24:])
    
    return live_df

# --- 3. Sidebar UI ---
st.sidebar.title("⚡ Grid Settings")
st.sidebar.markdown(f"**Location:** Karipur, KL\n\n**Lat/Lon:** `{LOCATION}`")

if st.sidebar.button("🔄 Refresh Live Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.write("### Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.progress(85, text="Confidence Level")

# --- 4. Main Application Logic ---
st.title("⚡ Electricity Demand Forecasting")
st.markdown(f"Real-time predictive analytics for tomorrow's grid load in **Karipur**.")

try:
    # Load Model and Data
    if not os.path.exists(ST_MODEL_PATH):
        st.error(f"Model file not found at {ST_MODEL_PATH}")
        st.stop()

    model = joblib.load(ST_MODEL_PATH)
    hist_df = pd.read_csv(ST_CSV_PATH)
    hist_df['Timestamp'] = pd.to_datetime(hist_df['Timestamp'])
    hist_df.set_index('Timestamp', inplace=True)

    weather_df = fetch_weather_forecast()

    if weather_df is not None:
        # Prepare Features
        live_features = prepare_features(weather_df.copy(), hist_df)
        
        # Predict
        X_live = live_features[model.feature_names_in_]
        predictions = model.predict(X_live)
        live_features['Predicted_Demand'] = predictions

        # --- 5. Professional Dashboard UI ---
        
        st.divider()
        st.subheader("💡 Tomorrow's Energy Outlook")
        
        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        peak_val = predictions.max()
        peak_time = live_features['Predicted_Demand'].idxmax().strftime('%I:%M %p')
        avg_temp = live_features['Temperature'].mean()
        total_mwh = predictions.sum()

        m1.metric("Peak Load", f"{peak_val:.1f} MW")
        m2.metric("Peak Time", peak_time)
        m3.metric("Avg Temp", f"{avg_temp:.1f} °C")
        m4.metric("Total Volume", f"{total_mwh:,.0f} MWh")

        st.divider()

        # Main Forecast Chart
        st.subheader("Hourly Predicted Load Profile")
        chart_df = live_features[['Predicted_Demand']].copy()
        chart_df.index = chart_df.index.strftime('%H:%M')
        st.area_chart(chart_df, color="#2E86C1", use_container_width=True)

        # Weather Correlation Section
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**Temperature & Humidity Trends**")
            st.line_chart(live_features[['Temperature', 'Humidity']])

        with col_right:
            st.write("**Grid Insights**")
            st.info(f"""
            - **Load Profile:** The grid expects a peak of **{peak_val:.2f} MW** around **{peak_time}**.
            - **Weather Factor:** Mean temperature is forecasted at **{avg_temp:.1f}°C**, which typically correlates with AC usage.
            - **Data Status:** Lag features synchronized with latest historical records from Desktop CSV.
            """)

        # Raw Data View
        with st.expander("📂 View Detailed Forecast Table"):
            st.dataframe(live_features[['Temperature', 'Humidity', 'Predicted_Demand']], use_container_width=True)

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.info("Check if the CSV path or Model path is still valid.")