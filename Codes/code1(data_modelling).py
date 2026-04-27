import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import holidays
data=pd.read_csv(r"C:\Users\smroo\OneDrive\Desktop\samrood_projects\weather\dataset\python machine learning model xgboost - electricity demand dataset.csv")
print(data.info())
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
print(data[['Temperature', 'Humidity', 'Demand']].describe())

#Checkin Missing Value

print(data.isnull().sum())
data.dropna(how='all', inplace=True)
cols_to_ffill = ['dayofweek', 'month', 'year', 'dayofyear']
data[cols_to_ffill] = data[cols_to_ffill].ffill()
data[['Temperature', 'Humidity']] = data[['Temperature', 'Humidity']].bfill()
data['Demand'] = data['Demand'].interpolate(method='time')

#Feature Engineering

data.insert(5, 'quarter', data.index.quarter)
data.insert(5, 'week_of_year', data.index.isocalendar().week.astype(int))
data.insert(7, 'is_weekend', data.index.dayofweek.isin([5, 6]).astype(int))
# Lagged Features (Previous day and previous week)
data['demand_lag_24hr'] = data['Demand'].shift(24)
data['demand_lag_168hr'] = data['Demand'].shift(168)
data['demand_rolling_mean_24h'] = data['Demand'].rolling(window=24).mean()
data['demand_rolling_std_24h'] = data['Demand'].rolling(window=24).std()
data.dropna(inplace=True)

#PLotting

data['Demand'].plot(figsize=(15, 6), title='Electricity Demand Over Time')
plt.ylabel('Demand in MW')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='hour', y='Demand')
plt.title('Demand by Hour of Day')
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

#modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
model_rf = RandomForestRegressor(n_estimators=500, max_depth=25, n_jobs=-1, random_state=42)
Y = data['Demand']
X = data.drop(columns=['Demand'])
#U can't use train_test_split on time series, B'coz it causes data leakage
X_train = X.loc[:'2023-12-31']
Y_train = Y.loc[:'2023-12-31']
X_test = X.loc['2024-01-01':]
Y_test = Y.loc['2024-01-01':]
# Training the model
# Using the same X_train and Y_train from the previous steps
model_rf.fit(X_train, Y_train)

#evaluation

predictions_rf = model_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(Y_test, predictions_rf))
mae_rf = mean_absolute_error(Y_test, predictions_rf)
print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest MAE: {mae_rf:.2f}")

# Plotting Feature Importance

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model_rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features (Random Forest)')
plt.show()

#Joblib

import joblib
joblib.dump(model_rf, 'electricity_RF_prediction_model.pkl')

