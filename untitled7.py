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
st.title("🌊 Aplikasi Prediksi Kualitas Air")
st.markdown("""
Gunakan aplikasi ini untuk memprediksi apakah air dapat diminum berdasarkan parameter kualitas air. 
Masukkan nilai-nilai parameter yang relevan di bawah ini untuk mendapatkan hasil prediksi.
""")

# Input Section
st.header("📊 Masukkan Parameter Kualitas Air")
with st.expander("💡 Petunjuk Penggunaan"):
    st.write("""
    - **pH Level**: Skala keasaman atau kebasaan air (0 - 14).
    - **Hardness (mg/L)**: Tingkat kekerasan air dalam miligram per liter.
    - **Dissolved Solids (mg/L)**: Jumlah zat padat terlarut dalam air.
    - **Chloramines (ppm)**: Konsentrasi kloramin dalam air.
    - **Sulfate (mg/L)**: Kandungan sulfat dalam miligram per liter.
    - **Conductivity (uS/cm)**: Kemampuan air menghantarkan listrik.
    - **Organic Carbon (ppm)**: Kandungan karbon organik dalam air.
    - **Trihalomethanes (ppb)**: Senyawa kimia dalam air.
    - **Turbidity (NTU)**: Kekeruhan air.
    """)

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    ph = st.slider("🌡️ pH Level", 0.0, 14.0, 7.0)
    hardness = st.slider("🧴 Hardness (mg/L)", 0.0, 300.0, 150.0)
    solids = st.slider("🧊 Dissolved Solids (mg/L)", 0.0, 50000.0, 20000.0)
    chloramines = st.slider("🧪 Chloramines (ppm)", 0.0, 12.0, 4.0)
    sulfate = st.slider("🔬 Sulfate (mg/L)", 0.0, 500.0, 250.0)

with col2:
    conductivity = st.slider("⚡ Conductivity (uS/cm)", 0.0, 800.0, 400.0)
    organic_carbon = st.slider("🌿 Organic Carbon (ppm)", 0.0, 30.0, 10.0)
    trihalomethanes = st.slider("🧴 Trihalomethanes (ppb)", 0.0, 120.0, 50.0)
    turbidity = st.slider("💧 Turbidity (NTU)", 0.0, 5.0, 1.0)

# Prediction button
if st.button("🔍 Cek Kualitas Air"):
    try:
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

        # Rename columns for scaler compatibility
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

        # Scale input data
        scaled_input = scaler.transform(input_data_english)

        # Make prediction
        prediction = random_forest_model.predict(scaled_input)[0]
        prediction_label = (
            "✅ **Dapat Diminum (Aman untuk Dikonsumsi)**"
            if prediction == 1 else
            "🚫 **Tidak Dapat Diminum (Tidak Aman untuk Dikonsumsi)**"
        )

        # Store results
        st.session_state.prediction_result = {
            "input_data": input_data,
            "prediction_label": prediction_label,
            "probabilities": random_forest_model.predict_proba(scaled_input)
            if hasattr(random_forest_model, "predict_proba") else None
        }
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Result Section
if st.session_state.prediction_result:
    st.header("📋 Hasil Prediksi")
    tab1, tab2 = st.tabs(["📈 Hasil Prediksi", "📊 Detail Probabilitas"])

    with tab1:
        st.write("### Input Data")
        st.table(st.session_state.prediction_result["input_data"])

        st.write("### Prediksi")
        st.success(st.session_state.prediction_result["prediction_label"])

    with tab2:
        if st.session_state.prediction_result["probabilities"] is not None:
            st.write("### Probabilitas Prediksi")
            st.bar_chart(pd.DataFrame(
                st.session_state.prediction_result["probabilities"],
                columns=["Tidak Dapat Diminum", "Dapat Diminum"]
            ))
        else:
            st.warning("Model ini tidak mendukung probabilitas.")
