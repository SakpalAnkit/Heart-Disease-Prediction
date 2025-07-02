import streamlit as st
import pandas as pd
import pickle as pkl
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load the model
try:
    model_random = pkl.load(open('model_random.pkl', 'rb'))
except Exception as e:
    st.error("Error loading the model. Please make sure 'model_random.pkl' is in the same folder.")
    st.stop()

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Minimal Custom CSS (optional)
st.markdown("""
<style>
.details-title { font-size:18px; color:#444; margin-bottom:15px; }
.error-msg { color: red; font-size:18px; font-weight: bold; }
.success-msg { color: green; font-size:18px; font-weight: bold; }
.health-metric { font-size:16px; margin-top:30px; }
.report-ready { font-size:16px; color: #006400; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>💓 Predict Heart Disease Risk 💓</h1>", unsafe_allow_html=True)
st.markdown('<p class="details-title">Enter the patient’s health details to predict the risk of heart disease.</p>', unsafe_allow_html=True)

# Input form
with st.form("heart_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Gender", ['Male', 'Female'])
        cp = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non Anginal Pain', 'Asymptomatic'])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['False', 'True'])
        restecg = st.selectbox("Resting ECG Result", ['Abnormality', 'Normal', 'Left ventricular hypertrophy'])
        exercise = st.selectbox("Exercise Induced Angina", ['No pain', 'Pain'])

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of ST Segment", ['Downsloping', 'Upsloping', 'Flat'])
        ca = st.selectbox("Number of Major Vessels Colored", [
            'Normal', 'One vessel colored', 'Two vessels colored', 'Three vessels colored', 'Four vessel colored'])

    thal = st.selectbox("Thalassemia", ['Reversible defect', 'Normal', 'Fixed Defect', 'Unknown'])

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Encoding categorical values
    sex_code = 1 if sex == "Male" else 0
    cp_map = {'Typical Angina': 3, 'Atypical Angina': 1, 'Non Anginal Pain': 2, 'Asymptomatic': 0}
    restecg_map = {'Abnormality': 0, 'Normal': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'No pain': 0, 'Pain': 1}
    slope_map = {'Downsloping': 0, 'Upsloping': 2, 'Flat': 1}
    ca_map = {'Normal': 0, 'One vessel colored': 1, 'Two vessels colored': 2, 'Three vessels colored': 3, 'Four vessel colored': 4}
    thal_map = {'Reversible defect': 2, 'Normal': 1, 'Fixed Defect': 0, 'Unknown': 3}

    input_data = pd.DataFrame([[  
        age, trestbps, chol, thalach, oldpeak,
        sex_code, cp_map[cp], int(fbs == "True"),
        restecg_map[restecg], exang_map[exercise],
        slope_map[slope], ca_map[ca], thal_map[thal]
    ]], columns=[
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
        'encoded_sex', 'encoded_cp', 'encoded_fbs',
        'encoded_restecg', 'encoded_exang', 'encoded_slope',
        'encoded_ca', 'encoded_thal'
    ])

    # Prediction with spinner
    with st.spinner("Analyzing health data..."):
        time.sleep(2)
        prediction = model_random.predict(input_data)[0]
        probability = model_random.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.markdown('<div class="error-msg">⚠️ This patient <strong>may have heart disease</strong>. Immediate consultation is recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-msg">✅ This patient is <strong>not likely to have heart disease</strong>.</div>', unsafe_allow_html=True)
        st.balloons()

    # Risk Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Heart Disease Risk"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "red" if prediction == 1 else "green"},
               'steps': [
                   {'range': [0, 0.5], 'color': "#c8f7c5"},
                   {'range': [0.5, 1], 'color': "#f9c0c0"}]
               }
    ))
    st.plotly_chart(fig)

    # Cholesterol bar
    st.markdown('<p class="health-metric">📊 Health Metric: Cholesterol</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.barh(["Cholesterol"], [chol], color="salmon" if chol > 200 else "seagreen")
    ax.axvline(x=200, color='black', linestyle='--', label='Normal Max')
    ax.set_xlabel("mg/dl")
    ax.legend()
    st.pyplot(fig)

    # Report download
    with st.spinner("Preparing your report..."):
        time.sleep(2)
        report_df = input_data.copy()
        report_df["prediction"] = prediction
        report_df["risk_percent"] = round(probability * 100, 2)
        csv = report_df.to_csv(index=False).encode('utf-8')

    st.download_button("📥 Download Report", data=csv, file_name="heart_risk_report.csv", mime='text/csv')
    st.markdown('<p class="report-ready">Your report is ready for download!</p>', unsafe_allow_html=True)
