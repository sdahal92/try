# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load trained pipeline (model + preprocessing)
model = joblib.load("client_retention_model.pkl")

# Streamlit setup
st.set_page_config(page_title="Client Retention + XAI", layout="centered")
st.title("🔄 Client Retention Predictor with XAI")

st.write("Use this tool to predict if a client will return — and explain why!")

# Collect user input (now in the main page)
st.header("📋 Client Input")
contact_method = st.selectbox("Contact Method", ['phone', 'email', 'in-person'])
household = st.selectbox("Household Type", ['single', 'family'])
preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
sex = st.selectbox("Sex", ['male', 'female'])
status = st.selectbox("Status", ['new', 'returning', 'inactive'])
season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
month = st.selectbox("Month", [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
latest_lang_english = st.selectbox("Latest Language is English", ['yes', 'no'])
age = st.slider("Age", 18, 100, 35)
dependents_qty = st.number_input("Number of Dependents", 0, 10, 1)
distance_km = st.number_input("Distance to Location (km)", 0.0, 50.0, 5.0)
num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)

# Predict on button click
if st.button("🔍 Predict"):
    # Prepare input for prediction
    input_df = pd.DataFrame([{
        'contact_method': contact_method,
        'household': household,
        'preferred_languages': preferred_language,
        'sex_new': sex,
        'status': status,
        'Season': season,
        'Month': month,
        'latest_language_is_english': latest_lang_english,
        'age': age,
        'dependents_qty': dependents_qty,
        'distance_km': distance_km,
        'num_of_contact_methods': num_of_contact_methods
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    conf_interval = f"{round(probability*100)}% ±10%"

    # Show result
    st.markdown("## 🧾 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Client is likely to return\nConfidence: {conf_interval}")
    else:
        st.warning(f"⚠️ Client may not return\nConfidence: {conf_interval}")

    # SHAP explanation
    st.markdown("---")
    st.subheader("🧠 Why This Prediction? (SHAP Explanation)")

    # Extract pipeline parts
    preprocessor = model.named_steps['preprocessing']
    classifier = model.named_steps['classifier']
    input_transformed = preprocessor.transform(input_df)

    # SHAP Explainer
    explainer = shap.Explainer(classifier, input_transformed)
    shap_values = explainer(input_transformed)

    # SHAP Waterfall Plot
    st.markdown("#### 📉 SHAP Waterfall Plot (Single Prediction)")
    fig1 = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    # SHAP Bar Plot
    st.markdown("#### 📊 SHAP Bar Chart (Top Features)")
    fig2 = plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig2)

    # Table of feature values
    st.markdown("#### 📋 Feature Contributions")
    feature_impact_df = pd.DataFrame({
        'Feature': shap_values.feature_names,
        'SHAP Value': shap_values.values[0],
        'Impact': np.where(shap_values.values[0] > 0, '↑ Increases Return', '↓ Decreases Return')
    }).sort_values(by='SHAP Value', key=np.abs, ascending=False)

    st.dataframe(feature_impact_df.head(10), use_container_width=True)
