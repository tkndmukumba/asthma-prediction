import streamlit as st
from joblib import load
import pandas as pd

# Load the saved model
model = load('asthma_predictor_orig.joblib')

# Title
st.title("Asthma Risk Prediction Tool")

# Input widgets organized into sections
st.sidebar.header("Patient Features")

# --- Demographics Section ---
st.header("1. Demographics")
col1, = st.columns(1)
with col1:
    Age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    Gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    Ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], 
                           format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
    EducationLevel = st.selectbox("Education Level", options=[0, 1, 2, 3],
                                format_func=lambda x: ["None", "High School", "Bachelor's", "Higher"][x])

    

# --- Lifestyle Factors ---
st.header("2. Lifestyle Factors")
col2, col3, col4 = st.columns(3)
with col2:
    BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
with col3:
    Smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Non-smoker" if x == 0 else "Smoker")
    PhysicalActivity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, 5.0)
    DietQuality = st.slider("Diet Quality (0-10 scale)", 0.0, 10.0, 7.0)
with col4:
    SleepQuality = st.slider("Sleep Quality (4-10 scale)", 4.0, 10.0, 7.0)
    

# --- Environmental and Allergy Factors ---
st.header("3. Environmental and Allergy Factors")
col5, col6 = st.columns(2)
with col5:
    PollutionExposure = st.slider("Pollution Exposure (0-10 scale)", 0.0, 10.0, 3.0)
    PollenExposure = st.slider("Pollen Exposure (0-10 scale)", 0.0, 10.0, 3.0)
    DustExposure = st.slider("Dust Exposure (0-10 scale)", 0.0, 10.0, 3.0)
    PetAllergy = st.selectbox("Pet Allergy", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col6:
    FamilyHistoryAsthma = st.selectbox("Family History of Asthma", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    HistoryOfAllergies = st.selectbox("History of Allergies", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# --- Medical Conditions ---
st.header("4. Medical History")
col7, col8 = st.columns(2)
with col7:
    Eczema = st.selectbox("Eczema", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    HayFever = st.selectbox("Hay Fever", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    GastroesophagealReflux = st.selectbox("GERD", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col8:
    LungFunctionFEV1 = st.number_input("FEV1 Lung Function Score", min_value=1.0, max_value=10.0, value=3.5)
    LungFunctionFVC = st.number_input("FVC Lung Function Score", min_value=1.0, max_value=10.0, value=3.5)

# --- Symptoms ---
st.header("5. Current Symptoms")
col9, col10 = st.columns(2)
with col9:
    Wheezing = st.selectbox("Wheezing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ShortnessOfBreath = st.selectbox("Shortness of Breath", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ChestTightness = st.selectbox("Chest Tightness", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col10:
    Coughing = st.selectbox("Coughing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    NighttimeSymptoms = st.selectbox("Nighttime Symptoms", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ExerciseInduced = st.selectbox("Exercise-Induced Symptoms", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Prediction button
if st.button("Predict Asthma Risk"):
    # Create input dictionary
    input_data = {
        'Age': Age,
        'Gender': Gender,
        'Ethnicity': Ethnicity,
        'BMI': BMI,
        'Smoking': Smoking,
        'PhysicalActivity': PhysicalActivity,
        'DietQuality': DietQuality,
        'SleepQuality': SleepQuality,
        'PollutionExposure': PollutionExposure,
        'PollenExposure': PollenExposure,
        'DustExposure': DustExposure,
        'PetAllergy': PetAllergy,
        'FamilyHistoryAsthma': FamilyHistoryAsthma,
        'HistoryOfAllergies': HistoryOfAllergies,
        'Eczema': Eczema,
        'HayFever': HayFever,
        'GastroesophagealReflux': GastroesophagealReflux,
        'LungFunctionFEV1': LungFunctionFEV1,
        'LungFunctionFVC': LungFunctionFVC,
        'Wheezing': Wheezing,
        'ShortnessOfBreath': ShortnessOfBreath,
        'ChestTightness': ChestTightness,
        'Coughing': Coughing,
        'NighttimeSymptoms': NighttimeSymptoms,
        'ExerciseInduced': ExerciseInduced
    }
    
    # Convert to DataFrame (ensure column order matches training)
    features = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]  # Probability of asthma (class 1)
    
    # Display results
    st.subheader("Results")
    st.success(f"Prediction: {'High Risk of Asthma' if prediction == 1 else 'Low Risk of Asthma'}")
    st.info(f"Confidence: {proba:.1%}")
    
    # Interpretation guide
    st.markdown("""
    **Interpretation Guide:**
    - **High Risk (≥50% probability)**: Consider consulting a pulmonologist
    - **Low Risk (<50% probability)**: Maintain regular check-ups
    """)

# Footer
st.markdown("---")
st.caption("Note: This tool is for screening purposes only. Always consult a healthcare professional for diagnosis.")