import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model and scaler
random_forest_model = joblib.load('random_forest.pkl')  # Ensure the model file is in the same directory
scaler = joblib.load('scaler.pkl')  # Ensure the scaler file is in the same directory

# Initialize session state
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Streamlit App Title
st.title("Water Quality Prediction Application")
st.write("Use this application to predict whether water is drinkable based on its quality parameters.")

# Input section
st.header("Enter Water Quality Parameters")
col1, col2 = st.columns(2)

with col1:
    ph = st.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness (mg/L)", 0.0, 300.0, 100.0)
    solids = st.slider("Dissolved Solids (mg/L)", 0.0, 50000.0, 20000.0)
    chloramines = st.slider("Chloramines (ppm)", 0.0, 12.0, 6.0)
    sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 200.0)

with col2:
    conductivity = st.slider("Conductivity (uS/cm)", 0.0, 800.0, 400.0)
    organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 15.0)
    trihalomethanes = st.slider("Trihalomethanes (ppb)", 0.0, 120.0, 60.0)
    turbidity = st.slider("Turbidity (NTU)", 0.0, 5.0, 2.5)

if st.button("Predict Water Quality"):
    # Combine user input into a DataFrame
    input_data = pd.DataFrame({
        'pH Level': [ph],
        'Hardness (mg/L)': [hardness],
        'Dissolved Solids (mg/L)': [solids],
        'Chloramines (ppm)': [chloramines],
        'Sulfate (mg/L)': [sulfate],
        'Conductivity (uS/cm)': [conductivity],
        'Organic Carbon (ppm)': [organic_carbon],
        'Trihalomethanes (ppb)': [trihalomethanes],
        'Turbidity (NTU)': [turbidity]
    })

    scaled_input = scaler.transform(input_data)

    # Make predictions
    prediction = random_forest_model.predict(scaled_input)[0]
    prediction_label = (
        "Drinkable (Safe for Consumption)"
        if prediction == 1 else
        "Non-Drinkable (Not Safe for Consumption)"
    )

    # Store result in session state
    st.session_state.prediction_result = {
        "input_data": input_data,
        "prediction_label": prediction_label,
        "probabilities": random_forest_model.predict_proba(scaled_input)
        if hasattr(random_forest_model, "predict_proba") else None
    }

# Result section
if st.session_state.prediction_result:
    st.header("Prediction Results")
    st.write("### Input Data")
    st.write(st.session_state.prediction_result["input_data"])

    st.write("### Prediction")
    st.success(st.session_state.prediction_result["prediction_label"])

    if st.session_state.prediction_result["probabilities"] is not None:
        st.write("### Prediction Probabilities")
        st.write(pd.DataFrame(
            st.session_state.prediction_result["probabilities"],
            columns=["Non-Drinkable", "Drinkable"]
        ))
