import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
try:
    from joblib import load
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "joblib"])
    from joblib import load
import time

# Load trained model
model = load("XGBmodel.joblib")  # Ensure this file exists in your directory

# Define Marsh score details
marsh_table = pd.DataFrame({
    "Category": ["Normal", "Mild", "Severe"],
    "Description": [
        "Normal Mucosa: No visible damage to villi.",
        "Increased immune activity or crypt hyperplasia with mild to moderate villous atrophy:",
        "Severe villous atrophy or total villous atrophy: Complete loss of villi leading to nutrient malabsorption."
    ],
    "Likelihood of Celiac Disease": [
        "Unlikely: Symptoms may suggest other conditions.",
        "Possible: Indicates early-stage, latent, or moderate celiac disease. Other conditions like infections or intolerances could contribute.",
        "Very likely: Severe damage strongly indicative of advanced celiac disease."
    ],
    "Severity": ["None", "Mild to Moderate", "Severe"],
    "Clinical Indications": [
        "No villous damage detected. Symptoms may suggest IBS or other conditions",
        "Mild or moderate damage to villi causing some nutrient absorption issues. Patient may experience mild fatigue, anemia, and digestive discomfort",
        "Complete or near-complete damage to villi. Patient may experience severe malnutrition, chronic diarrhea, and weight loss."
    ]
})

# Updated Marsh Mapping for Three Categories
marsh_mapping = {
    0: "Normal Category(Marsh 0)",
    1: "Mild Category(Marsh 1, Marsh 2)",
    2: "Severe Category(Marsh 3a, 3b, 3c)"
}


# Streamlit Theme Configuration
st.set_page_config(page_title="Celiac Prediction Tool", page_icon="🔮", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #85daed;
            color: #e0f2fe;
            font-family: 'Roboto', sans-serif;
        }
        .stTitle, .stHeader, .stSubheader {
            color: #f472b6;
            text-shadow: 0px 0px 5px #f472b6;
        }
        .stButton>button {
            background-color: #14b8a6;
            color: #fff;
            border-radius: 12px;
            font-size: 16px;
        }
        .stDataFrame {
            border: 2px solid #10b981;
        }
        hr {
            border: 1px solid #4ade80;
        }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.title("🧪 Celiac Disease Prediction and Severity for Diabetic Patients Using XGBoost")
st.write("### Input patient data to predict the likelihood and severity of Celiac Disease using a trained XGBoost machine learning model .")
st.write("<hr>", unsafe_allow_html=True)

# Input Fields
st.subheader("📝 Enter Patient Data")
diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2"])
short_stature = st.radio("Short Stature?", ["DSS", "PSS", "Variant"])
sticky_stool = st.radio("Sticky Stool?", ["Yes", "No"])
weight_loss = st.radio("Weight Loss?", ["Yes", "No"])
iga = st.number_input("Enter IgA Levels (g/L):", min_value=0.0, max_value=500.0, step=0.1)
igg = st.number_input("Enter IgG Levels (g/L):", min_value=0.0, max_value=500.0, step=0.1)

# Encode inputs
diabetes_encoded = 0 if diabetes_type == "Type 1" else 1
stature_encoded = {"DSS": 0, "PSS": 1, "Variant": 2}[short_stature]
sticky_encoded = 1 if sticky_stool == "Yes" else 0
weight_encoded = 1 if weight_loss == "Yes" else 0

# Prediction Button
if st.button("Predict"):
    # Prepare input
    user_data = pd.DataFrame({
        'Diabetes Type': [diabetes_encoded],
        'Short_Stature': [stature_encoded],
        'Sticky_Stool': [sticky_encoded],
        'Weight_loss': [weight_encoded],
        'IgA': [iga],
        'IgG': [igg]
    })

    # Show loader animation
    with st.spinner("🤖 Analyzing data and predicting..."):
        time.sleep(2)  # Simulate loading time
        prediction = model.predict(user_data)[0]

    # Display Results
    st.success(f"🎯 **Predicted Marsh Score Category**: {marsh_mapping[prediction]}")

    # Display Clinical Details
    st.subheader("🔎 Clinical Insights")
    row = marsh_table.iloc[prediction]
    table = pd.DataFrame(row).T.reset_index(drop=True)  # Reset the index
    table.index = ['']
    styled_table = table.style.set_table_styles([
    {"selector": "th", "props": [("text-align", "left"), ("width", "200px")]},
    {"selector": "td", "props": [("text-align", "left"), ("width", "500px")]}
])
    st.table(styled_table)
    #row = marsh_table.iloc[prediction]
    #table = pd.DataFrame(row).T  # Create a table
    #st.table(table)

# Footer
st.write("<hr>", unsafe_allow_html=True)
st.write("🚨 *Note: This tool is not a replacement for medical advice. Please consult a healthcare professional.*")
