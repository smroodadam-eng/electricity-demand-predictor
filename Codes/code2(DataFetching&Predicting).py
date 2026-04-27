import pandas as pd
import numpy as np
import joblib
import urllib.request  # Corrected import
import json
# 1. SETUP PARAMETERS
MODEL_PATH = 'electricity_RF_prediction_model.pkl'
CSV_PATH = r"C:\Users\smroo\OneDrive\Desktop\samrood_projects\weather\dataset\python machine learning model xgboost - electricity demand dataset.csv"
API_KEY = "8H37ZY74J5X2EFZLNJ9LUWKKK"
LOCATION = "11.1457,75.9643"
# 2. FETCH LATEST KNOWN DATA FROM CSV (For Lags)
print("Loading historical data for lag features...")
historical_df = pd.read_csv(CSV_PATH)
historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp'])
historical_df.set_index('Timestamp', inplace=True)
historical_df.sort_index(inplace=True)
# Get the last 168 hours of demand to calculate rolling stats and lags
last_known_demand = historical_df['Demand'].tail(168).values
last_24h_val = last_known_demand[-1] # Lag 1 for a daily model, or tail(24) for hourly
last_rolling_mean = np.mean(last_known_demand[-24:])
last_rolling_std = np.std(last_known_demand[-24:])
# 3. FETCH TOMORROW'S WEATHER
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}/tomorrow?unitGroup=metric&contentType=json&key={API_KEY}"

with urllib.request.urlopen(url) as response:
    raw_data = json.loads(response.read().decode())

# Parse hourly forecast
forecast_list = []
for hr in raw_data['days'][0]['hours']:
    forecast_list.append({
        'Timestamp': raw_data['days'][0]['datetime'] + ' ' + hr['datetime'],
        'Temperature': hr['temp'],
        'Humidity': hr['humidity']
    })

live_df = pd.DataFrame(forecast_list)
live_df['Timestamp'] = pd.to_datetime(live_df['Timestamp'])
live_df.set_index('Timestamp', inplace=True)

# 4. FEATURE ENGINEERING (Must match Phase 1 EXACTLY)
live_df['hour'] = live_df.index.hour
live_df['dayofweek'] = live_df.index.dayofweek
live_df['month'] = live_df.index.month
live_df['year'] = live_df.index.year
live_df['dayofyear'] = live_df.index.dayofyear
live_df['quarter'] = live_df.index.quarter
live_df['week_of_year'] = live_df.index.isocalendar().week.astype(int)
live_df['is_weekend'] = live_df.index.dayofweek.isin([5, 6]).astype(int)

# Fill Lags using the CSV data we pulled earlier
live_df['demand_lag_24hr'] = last_known_demand[-24] # Demand from exactly 24h ago
live_df['demand_lag_168hr'] = last_known_demand[0]  # Demand from exactly 168h ago
live_df['demand_rolling_mean_24h'] = last_rolling_mean
live_df['demand_rolling_std_24h'] = last_rolling_std

# 5. LOAD MODEL & PREDICT
print("Running predictions...")
model = joblib.load(MODEL_PATH)
# 1. Get the exact list of features the model was trained on
expected_features = model.feature_names_in_.tolist()

# 2. Reorder live_df to match that exact list
# This also drops any extra columns (like 'Timestamp') that aren't features
live_df_final = live_df[expected_features]

# 3. Now run the prediction using the aligned dataframe
print("Running aligned predictions...")
predictions = model.predict(live_df_final)

# 6. DISPLAY RESULTS
live_df['Predicted_Demand'] = predictions
print("\n--- Predictions for Tomorrow ---")
print(live_df[['Temperature', 'Predicted_Demand']].head(12))

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Get the last 24 hours of REAL data from your CSV for comparison
past_24h = historical_df.tail(24)

# 2. Set up the plot style
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# 3. Plot Past Actual Demand (Blue)
plt.plot(past_24h.index, past_24h['Demand'],label='Past 24h (Actual)', color='#2E86C1', linewidth=2, marker='o')
# 4. Plot Tomorrow's Predicted Demand (Orange/Red)
plt.plot(live_df.index, live_df['Predicted_Demand'],label='Tomorrow (Predicted)', color='#E67E22', linewidth=3, linestyle='--', marker='s')
# 5. Add Labels and Title
plt.title(f"Electricity Demand Forecast: {LOCATION} (Karipur, Kerala)", fontsize=16, pad=20)
plt.xlabel("Time (Hourly)", fontsize=12)
plt.ylabel("Demand (MW)", fontsize=12)
plt.legend(loc='upper left', fontsize=12)
# 6. Format the X-axis for readability
plt.xticks(rotation=45)
plt.tight_layout()
# 7. Save and Show
plt.savefig('demand_forecast_plot.png')
print("Plot saved as 'demand_forecast_plot.png'")
plt.show()