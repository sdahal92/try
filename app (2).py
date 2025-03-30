# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load trained model (pipeline: preprocessing + classifier)
model = joblib.load("client_retention_model.pkl")

st.set_page_config(page_title="Client Retention + XAI", layout="wide")
st.title("🔄 Client Retention Predictor with XAI")

st.write("This tool predicts if a client will return, and explains the key features behind that prediction.")

# Input form
with st.form("prediction_form"):
    st.sidebar.header("📋 Client Input")
    contact_method = st.sidebar.selectbox("Contact Method", ['phone', 'email', 'in-person'])
    household = st.sidebar.selectbox("Household Type", ['single', 'family'])
    preferred_language = st.sidebar.selectbox("Preferred Language", ['english', 'other'])
    sex = st.sidebar.selectbox("Sex", ['male', 'female'])
    status = st.sidebar.selectbox("Status", ['new', 'returning', 'inactive'])
    season = st.sidebar.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
    month = st.sidebar.selectbox("Month", [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    latest_lang_english = st.sidebar.selectbox("Latest Language is English", ['yes', 'no'])
    age = st.sidebar.slider("Age", 18, 100, 35)
    dependents_qty = st.sidebar.number_input("Number of Dependents", 0, 10, 1)
    distance_km = st.sidebar.number_input("Distance to Location (km)", 0.0, 50.0, 5.0)
    num_of_contact_methods = st.sidebar.slider("Number of Contact Methods", 1, 5, 2)
    submitted = st.form_submit_button("🔍 Predict")

# Process input & predict
if submitted:
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

    # Simulated confidence interval
    conf_interval = f"{round(probability*100)}% ±10%"

    st.markdown("---")
    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.success(f"✅ Client is likely to return\nConfidence: {conf_interval}")
    else:
        st.warning(f"⚠️ Client may not return\nConfidence: {conf_interval}")

    # SHAP explanation
    st.markdown("---")
    st.subheader("🧠 Why This Prediction? (SHAP Explanation)")

    # Extract pipeline components
    preprocessor = model.named_steps['preprocessing']
    classifier = model.named_steps['classifier']

    # Preprocess input
    input_transformed = preprocessor.transform(input_df)

    # SHAP explainer
    explainer = shap.Explainer(classifier, input_transformed)
    shap_values = explainer(input_transformed)

    # SHAP waterfall
    st.markdown("#### 🔍 SHAP Waterfall Plot (Individual Prediction)")
    fig1 = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    # SHAP bar chart
    st.markdown("#### 📊 SHAP Bar Chart (Most Impactful Features)")
    fig2 = plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig2)

    # SHAP value table
    st.markdown("#### 📋 Feature Contributions (Table)")
    feature_impact_df = pd.DataFrame({
        'Feature': shap_values.feature_names,
        'SHAP Value': shap_values.values[0],
        'Impact Direction': np.where(shap_values.values[0] > 0, '↑ Increases Return', '↓ Decreases Return')
    }).sort_values('SHAP Value', key=np.abs, ascending=False)

    st.dataframe(feature_impact_df.head(10), use_container_width=True)
