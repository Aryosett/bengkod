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
st.title("Aplikasi Prediksi Kualitas Air")
st.write("Gunakan aplikasi ini untuk memprediksi apakah air dapat diminum berdasarkan parameter kualitas air.")

# Input section
st.header("Masukkan Parameter Kualitas Air")

ph = st.slider("Tingkat pH", 0.0, 14.0, 7.0)
hardness = st.slider("Kekerasan (mg/L)", 0.0, 300.0, 100.0)
solids = st.slider("Padatan Terlarut (mg/L)", 0.0, 50000.0, 20000.0)
chloramines = st.slider("Kloramin (ppm)", 0.0, 12.0, 6.0)
sulfate = st.slider("Sulfat (mg/L)", 0.0, 500.0, 200.0)
conductivity = st.slider("Konduktivitas (uS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.slider("Karbon Organik (ppm)", 0.0, 30.0, 15.0)
trihalomethanes = st.slider("Trihalometana (ppb)", 0.0, 120.0, 60.0)
turbidity = st.slider("Kekeruhan (NTU)", 0.0, 5.0, 2.5)

if st.button("Prediksi Kualitas Air"):
    # Combine user input into a DataFrame
    input_data_indonesia = pd.DataFrame({
        'Tingkat pH': [ph],
        'Kekerasan (mg/L)': [hardness],
        'Padatan Terlarut (mg/L)': [solids],
        'Kloramin (ppm)': [chloramines],
        'Sulfat (mg/L)': [sulfate],
        'Konduktivitas (uS/cm)': [conductivity],
        'Karbon Organik (ppm)': [organic_carbon],
        'Trihalometana (ppb)': [trihalomethanes],
        'Kekeruhan (NTU)': [turbidity]
    })

    input_data_english = input_data_indonesia.rename(columns={
        'Tingkat pH': 'ph',
        'Kekerasan (mg/L)': 'Hardness',
        'Padatan Terlarut (mg/L)': 'Solids',
        'Kloramin (ppm)': 'Chloramines',
        'Sulfat (mg/L)': 'Sulfate',
        'Konduktivitas (uS/cm)': 'Conductivity',
        'Karbon Organik (ppm)': 'Organic_carbon',
        'Trihalometana (ppb)': 'Trihalomethanes',
        'Kekeruhan (NTU)': 'Turbidity'
    })

    scaled_input = scaler.transform(input_data_english)

    # Make predictions
    prediction = random_forest_model.predict(scaled_input)[0]
    prediction_label = (
        "Dapat Diminum (Aman untuk Dikonsumsi)"
        if prediction == 1 else
        "Tidak Dapat Diminum (Tidak Aman untuk Dikonsumsi)"
    )

    # Store result in session state
    st.session_state.prediction_result = {
        "input_data": input_data_indonesia,
        "prediction_label": prediction_label,
        "probabilities": random_forest_model.predict_proba(scaled_input)
        if hasattr(random_forest_model, "predict_proba") else None
    }

# Result section
if st.session_state.prediction_result:
    st.header("Hasil Prediksi")
    st.write("### Data yang Dimasukkan")
    st.write(st.session_state.prediction_result["input_data"])

    st.write("### Prediksi")
    st.success(st.session_state.prediction_result["prediction_label"])

    if st.session_state.prediction_result["probabilities"] is not None:
        st.write("### Probabilitas Prediksi")
        st.write(pd.DataFrame(
            st.session_state.prediction_result["probabilities"],
            columns=["Tidak Dapat Diminum", "Dapat Diminum"]
        ))
