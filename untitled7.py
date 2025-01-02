import streamlit as st
import pandas as pd
import joblib
import random

# Load the Random Forest model and scaler
random_forest_model = joblib.load('random_forest.pkl')  # Ensure the model file is in the same directory
scaler = joblib.load('scaler.pkl')  # Ensure the scaler file is in the same directory

# Initialize session state
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Streamlit App Title
st.title("Water Quality Prediction App")
st.write("Use this app to predict whether the water is safe to drink based on water quality parameters.")

# Input section
st.header("Enter Water Quality Parameters")
col1, col2 = st.columns(2)

with col1:
    ph = random.uniform(0.0, 14.0)
    hardness = random.uniform(0.0, 300.0)
    solids = random.uniform(0.0, 50000.0)
    chloramines = random.uniform(0.0, 12.0)
    sulfate = random.uniform(0.0, 500.0)

with col2:
    conductivity = random.uniform(0.0, 800.0)
    organic_carbon = random.uniform(0.0, 30.0)
    trihalomethanes = random.uniform(0.0, 120.0)
    turbidity = random.uniform(0.0, 5.0)

if st.button("Predict Water Quality"):
    # Combine user input into a DataFrame
    input_data_english = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    scaled_input = scaler.transform(input_data_english)

    # Make predictions
    prediction = random_forest_model.predict(scaled_input)[0]
    prediction_label = (
        "Safe to Drink (Safe for Consumption)"
        if prediction == 1 else
        "Not Safe to Drink (Not Safe for Consumption)"
    )

    # Store result in session state
    st.session_state.prediction_result = {
        "input_data": input_data_english,
        "prediction_label": prediction_label,
        "probabilities": random_forest_model.predict_proba(scaled_input)
        if hasattr(random_forest_model, "predict_proba") else None
    }

# Result section
if st.session_state.prediction_result:
    st.header("Prediction Results")
    st.write("### Entered Data")
    st.write(st.session_state.prediction_result["input_data"])

    st.write("### Prediction")
    st.success(st.session_state.prediction_result["prediction_label"])

    if st.session_state.prediction_result["probabilities"] is not None:
        st.write("### Prediction Probabilities")
        st.write(pd.DataFrame(
            st.session_state.prediction_result["probabilities"],
            columns=["Not Safe to Drink", "Safe to Drink"]
        ))
