import streamlit as st
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

# Load the Random Forest model and scaler
try:
    random_forest_model = joblib.load('random_forest.pkl')  # Pastikan file model tersedia
    scaler = joblib.load('scaler.pkl')  # Pastikan file scaler tersedia
except FileNotFoundError as e:
    st.error(f"Error: {e}. Pastikan file model dan scaler tersedia di direktori yang sama.")
    st.stop()

# Initialize session state
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Streamlit App Title
st.title(":droplet: Aplikasi Prediksi Kualitas Air")
st.markdown(
    "Gunakan aplikasi ini untuk memprediksi apakah air dapat diminum berdasarkan parameter kualitas air.\n\n"
    "**Instruksi:** Geser slider untuk mengatur parameter kualitas air, lalu klik tombol 'Cek Kualitas Air' untuk melihat hasil prediksi."
)

# Input section
st.sidebar.header("Masukkan Parameter Kualitas Air")
col1, col2 = st.columns(2)

with col1:
    ph = st.sidebar.slider("pH Level", 0.0, 14.0, 10.0)
    hardness = st.sidebar.slider("Hardness (mg/L)", 0.0, 300.0, 50.0)
    solids = st.sidebar.slider("Dissolved Solids (mg/L)", 0.0, 50000.0, 35000.0)
    chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 12.0, 8.0)
    sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 400.0)

with col2:
    conductivity = st.sidebar.slider("Conductivity (uS/cm)", 0.0, 800.0, 700.0)
    organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 20.0)
    trihalomethanes = st.sidebar.slider("Trihalomethanes (ppb)", 0.0, 120.0, 50.0)
    turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 5.0, 2.5)

if st.button("Cek Kualitas Air"):
    try:
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

        # Rename columns to match the scaler's feature names
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

        # Scale the input data
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
            "input_data": input_data,
            "prediction_label": prediction_label,
            "probabilities": random_forest_model.predict_proba(scaled_input)
            if hasattr(random_forest_model, "predict_proba") else None
        }
    except NotFittedError as e:
        st.error(f"Error: Model belum siap digunakan. Pastikan model telah dilatih. ({e})")

# Result section
if st.session_state.prediction_result:
    st.header(":mag: Hasil Prediksi")
    st.write("### Input Data")
    st.dataframe(st.session_state.prediction_result["input_data"])

    st.write("### Prediksi")
    st.success(st.session_state.prediction_result["prediction_label"])

    if st.session_state.prediction_result["probabilities"] is not None:
        st.write("### Probabilitas Prediksi")
        st.bar_chart(pd.DataFrame(
            st.session_state.prediction_result["probabilities"],
            columns=["Tidak Dapat Diminum", "Dapat Diminum"]
        ))

# Add footer
st.markdown(
    "---\n"
    "### Tentang Aplikasi\n"
    "Aplikasi ini dirancang untuk membantu dalam menentukan kualitas air berdasarkan parameter tertentu. \n"
    "**Pengembang:** Tim Kualitas Air | **Versi:** 1.0"
)
