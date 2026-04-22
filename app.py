import streamlit as st
import numpy as np
import joblib
import os

# Load model + scaler
# This ensures the app looks in its own folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# UI
st.title("AAMI Standard ECG Anomaly Detection")
st.write("Enter 187 ECG values (comma separated)")

# Input
input_data = st.text_area("ECG Signal Input")

# Prediction
if st.button("Predict"):
    try:
        values = [float(x) for x in input_data.split(",")]

        if len(values) != 187:
            st.error("Please enter exactly 187 values")
        else:
            data = np.array(values).reshape(1, -1)
            data = scaler.transform(data)

            pred = model.predict(data)[0]

            classes = {
                0: "Normal (N)",
                1: "Supraventricular (S)",
                2: "Ventricular (V)",
                3: "Fusion (F)",
                4: "Unknown (Q)"
            }

            st.success(f"Prediction: {classes[pred]}")

    except:
        st.error("Invalid input")