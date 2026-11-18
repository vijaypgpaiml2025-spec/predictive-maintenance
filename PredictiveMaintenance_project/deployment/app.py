import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Correct path to the trained model serialized inside the deployment folder
model_path = os.path.join(os.path.dirname(__file__), "best_engine_failure_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance Prediction
st.title("⚙️ Engine Failure Predictive Maintenance Dashboard")
st.write("Predict engine failure status based on real-time operational sensor parameters.")

# Input fields based on data dictionary
engine_rpm = st.number_input("Engine RPM", value=1500.0, step=10.0)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", value=3.5, step=0.1)
fuel_pressure = st.number_input("Fuel Pressure (bar)", value=5.0, step=0.1)
coolant_pressure = st.number_input("Coolant Pressure (bar)", value=2.5, step=0.1)
lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", value=85.0, step=0.5)
coolant_temp = st.number_input("Coolant Temperature (°C)", value=80.0, step=0.5)

if st.button("Predict Engine Status"):
    # 1. Align column names with the standard snake_case expected by our preparation/model pipeline
    input_data = pd.DataFrame({
        "engine_rpm": [engine_rpm],
        "lub_oil_pressure": [lub_oil_pressure],
        "fuel_pressure": [fuel_pressure],
        "coolant_pressure": [coolant_pressure],
        "lub_oil_temp": [lub_oil_temp],
        "coolant_temp": [coolant_temp]
    })

    # 2. Add engineered features identical to the training preparation phase
    input_data['pressure_ratio'] = input_data['lub_oil_pressure'] / (input_data['fuel_pressure'] + 1e-5)
    input_data['temp_differential'] = input_data['coolant_temp'] - input_data['lub_oil_temp']

    # Make prediction using the loaded model pipeline (which contains the scaler and classifier)
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 Warning: Engine Fault / Maintenance Required!")
    else:
        st.success("✅ Engine Operation Normal.")
