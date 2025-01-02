import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model and scaler
random_forest_model = joblib.load('random_forest.pkl')  # Ensure the model file is in the same directory
scaler = joblib.load('scaler.pkl')  # Ensure the scaler file is in the same directory

# Initialize session state for tab management
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Input Data"

# Streamlit App Title
st.title("Aplikasi Prediksi Kualitas Air")
st.write("Gunakan aplikasi ini untuk memprediksi apakah air dapat diminum berdasarkan parameter kualitas air.")

# Tab Layout
input_tab, result_tab = st.tabs(["Input Data", "Hasil Prediksi"])

# Sidebar for user inputs
with input_tab:
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

    # Add a submit button
    if st.button("Kirim dan Prediksi"):
        # Store input data in session state
        st.session_state.input_data = input_data_indonesia

        # Map feature names from Bahasa Indonesia to English
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

        # Normalize user input using the pre-trained scaler
        scaled_input = scaler.transform(input_data_english)

        # Make predictions using the Random Forest model
        prediction = random_forest_model.predict(scaled_input)[0]
        prediction_label = "Dapat Diminum (Aman untuk Dikonsumsi)" if prediction == 1 else "Tidak Dapat Diminum (Tidak Aman untuk Dikonsumsi)"
        st.session_state.prediction_label = prediction_label

        # Store prediction probabilities
        if hasattr(random_forest_model, "predict_proba"):
            probabilities = random_forest_model.predict_proba(scaled_input)
            st.session_state.probabilities = probabilities

        # Switch to the result tab
        st.session_state.active_tab = "Hasil Prediksi"

# Handle result tab
if st.session_state.active_tab == "Hasil Prediksi":
    with result_tab:
        st.header("Hasil Prediksi")

        # Display user input data
        st.write("### Data yang Dimasukkan")
        st.write(st.session_state.get("input_data", "Data tidak tersedia."))

        # Display prediction result
        st.write("### Prediksi")
        st.success(f"**{st.session_state.get('prediction_label', 'Belum ada hasil prediksi.')}**")

        # Display prediction probabilities if available
        if "probabilities" in st.session_state:
            st.write("### Probabilitas Prediksi")
            st.write(pd.DataFrame(st.session_state.probabilities, columns=["Tidak Dapat Diminum", "Dapat Diminum"]))
