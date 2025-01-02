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
st.title("\ud83d\udca7 Aplikasi Prediksi Kualitas Air")
st.write("Gunakan aplikasi ini untuk memprediksi apakah air dapat diminum berdasarkan parameter kualitas air.")

# Input section
st.sidebar.header("Masukkan Parameter Kualitas Air")

ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 300.0, 150.0)
solids = st.sidebar.slider("Dissolved Solids (mg/L)", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 12.0, 4.0)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0)
conductivity = st.sidebar.slider("Conductivity (uS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 15.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes (ppb)", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 5.0, 2.5)

if st.sidebar.button("\ud83c\udf10 Cek Kualitas Air"):
    # Combine user input into a DataFrame with correct column names
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

    # Rename columns to match the scaler's feature names (ensure these are what the scaler expects)
    input_data_english = input_data.rename(columns={
        'pH Level': 'ph',
        'Hardness (mg/L)': 'Hardness',
        'Dissolved Solids (mg/L)': 'Solids',
        'Chloramines (ppm)': 'Chloramines',
        'Sulfate (mg/L)': 'Sulfate',
        'Conductivity (uS/cm)': 'Conductivity',
        'Organic Carbon (ppm)': 'Organic_carbon',
        'Trihalomethanes (ppb)': 'Trihalomethanes',
        'Turbidity (NTU)': 'Turbidity'
    })

    # Ensure the input data columns match the scaler's expected feature names
    scaled_input = scaler.transform(input_data_english)

    # Make predictions
    prediction = random_forest_model.predict(scaled_input)[0]
    prediction_label = (
        "\ud83c\udf0a Dapat Diminum (Aman untuk Dikonsumsi)"
        if prediction == 1 else
        "\u26a0\ufe0f Tidak Dapat Diminum (Tidak Aman untuk Dikonsumsi)"
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
    st.header("\ud83d\udcca Hasil Prediksi")
    st.write("### Input Data")
    st.write(st.session_state.prediction_result["input_data"])

    st.write("### Prediksi")
    st.success(st.session_state.prediction_result["prediction_label"])

    if st.session_state.prediction_result["probabilities"] is not None:
        st.write("### Probabilitas Prediksi")
        st.write(pd.DataFrame(
            st.session_state.prediction_result["probabilities"],
            columns=["Tidak Dapat Diminum", "Dapat Diminum"]
        ))
