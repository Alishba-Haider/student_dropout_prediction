import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("dropout_model.pkl")
scaler = joblib.load("dropout_scaler.pkl")

# Load the feature columns used during training
feature_columns = joblib.load("feature_columns.pkl")


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🎓 Student Dropout Risk Prediction")

st.write(
    "Enter student information below to predict the probability "
    "of dropout and determine the student's risk level."
)


# ==========================================
# STUDENT INPUT
# ==========================================

st.header("Student Information")

age = st.number_input(
    "Age",
    min_value=15,
    max_value=60,
    value=20
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

gpa = st.number_input(
    "GPA",
    min_value=0.0,
    max_value=4.0,
    value=2.5
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Dropout Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "GPA": [gpa]
    })

    # Make sure columns match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Convert to percentage
    probability_percent = probability * 100


    # ==========================================
    # RISK CATEGORY
    # ==========================================

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"


    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    st.subheader("Prediction Result")

    st.write(
        f"**Dropout Probability:** {probability_percent:.2f}%"
    )

    st.write(
        f"**Risk Category:** {risk}"
    )