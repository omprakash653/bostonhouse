import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Boston House Price Predictor", layout="centered")

st.title("🏠 Boston House Price Prediction")
st.write("Enter the feature values below to predict the house price.")

# Input fields
CRIM = st.number_input("CRIM (Per capita crime rate)", value=0.1)
ZN = st.number_input("ZN (Residential land zoned)", value=0.0)
INDUS = st.number_input("INDUS (Non-retail business acres)", value=11.0)
CHAS = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
NOX = st.number_input("NOX (Nitric oxides concentration)", value=0.5)
RM = st.number_input("RM (Average number of rooms)", value=6.0)
AGE = st.number_input("AGE (Proportion of owner-occupied units built before 1940)", value=65.0)
DIS = st.number_input("DIS (Distance to employment centers)", value=4.0)
RAD = st.number_input("RAD (Accessibility to radial highways)", value=5)
TAX = st.number_input("TAX (Property tax rate)", value=300)
PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio)", value=18.0)
B = st.number_input("B (1000(Bk - 0.63)^2)", value=390.0)
LSTAT = st.number_input("LSTAT (% lower status population)", value=12.0)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[
            CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
            DIS, RAD, TAX, PTRATIO, B, LSTAT
        ]],
        columns=[
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
            'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")