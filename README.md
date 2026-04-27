Project Overview: 

This tool is designed to help grid operators and energy analysts visualize the Daily Load Profile.By combining historical 
demand "lags" with real-time weather variables (Temperature and Humidity), the model can anticipate spikes in energy consumption before they happen.

Features:

Live Weather Integration: Fetches hourly forecasts via the Visual Crossing Timeline API.Predictive Modeling: Utilizes a Random Forest Regressor ($n=500$) trained on multi-year demand data.Interactive Dashboard: * KPI Cards: Instant view of Peak Load, Peak Timing, and Average Temperature.Area Charts: High-fidelity visualization of the forecasted demand curve.Weather Analysis: Real-time correlation charts between temperature trends and energy needs.Large File Handling: Optimized using Git LFS to manage the 1.6GB model architecture.

Repository Structure:

.
├── Codes/
│   ├── app.py                            # Streamlit Dashboard & Inference Logic
│   └── electricity_RF_prediction_model.pkl # Trained Random Forest Model (LFS)
├── .gitattributes                        # Git LFS configuration
├── .gitignore                            # Excludes large binaries & venv
└── README.md                             # Project Documentation

Setup & Installation:

1. Prerequisites:
Python 3.9+
A Visual Crossing API Key (https://www.visualcrossing.com/weather-data/)
2. Installation:
#clone the repo
git clone https://github.com/smroodadam-eng/electricity-demand-predictor.git
cd electricity-demand-predictor

#create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

#Install dependencies
pip install streamlit pandas numpy scikit-learn joblib
3. Data Configuration
Update the ST_CSV_PATH in app.py to point to your local dataset location, or move your dataset into the Codes folder for portable use.

Methodology:
The model captures the Diurnal Cycle of energy usage. It calculates demand (P) as a function of temporal features and weather drivers:
P_{tomorrow} = f(Hour, DayOfWeek, Temp, Humidity, Demand_{lag24h}

Running the Dashboard:

Navigate to the Codes directory and launch the Streamlit app:
streamlit run app.py

Contact:

Samrood Project Link: https://github.com/smroodadam-eng/electricity-demand-predictor
