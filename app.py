import streamlit as st
import pandas as pd
import joblib

model = joblib.load('telco_churn_model.joblib')
scaler = joblib.load('telco_churn_scaler.joblib')

# Load original data as a template (To maintain the column structure)
df_template = pd.read_csv('telco_churn.csv')
df_template.drop(['customerID', 'Churn'], axis=1, inplace=True, errors='ignore')

# Page and UI Settings
st.set_page_config(page_title="Churn Prediction", page_icon="📞", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.markdown("End-to-End Machine Learning Portfolio Project.")

st.sidebar.header("Enter Customer Information")

# Get user input from sidebar
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", value=70.0)

# Logic Update: Automatically calculate Total Charges based on Tenure and Monthly Charges
TotalCharges = tenure * MonthlyCharges
st.sidebar.markdown(f"**Calculated Total Charges:** ${TotalCharges:.2f}")

Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Predict Button and Analysis
if st.button("Calculate Risk 🚀"):
    with st.spinner("AI is analyzing the customer..."):

        # Convert user input to dictionary
        input_dict = {
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'Contract': Contract,
            'InternetService': InternetService
        }

        # Fill missing columns from the template's first row to keep the 30-column structure intact
        for col in df_template.columns:
            if col not in input_dict:
                input_dict[col] = df_template.iloc[0][col]

        df_input = pd.DataFrame([input_dict])

        # Combine with original template and apply One-Hot Encoding
        df_combined = pd.concat([df_template, df_input], ignore_index=True)
        df_combined['TotalCharges'] = pd.to_numeric(df_combined['TotalCharges'], errors='coerce').fillna(0)
        df_encoded = pd.get_dummies(df_combined, drop_first=True)

        # Extract the new customer row (the last one we just appended)
        X_new_customer = df_encoded.iloc[[-1]]

        # Scale the features
        X_new_customer_scaled = scaler.transform(X_new_customer)

        # Model Prediction
        prediction = model.predict(X_new_customer_scaled)
        probability = model.predict_proba(X_new_customer_scaled)[0][1]

        # Display Results
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"⚠️ **HIGH RISK!** Probability of churn: **{probability * 100:.1f}%**")
            st.info("💡 Recommendation: Offer a 1 or 2-year contract immediately to retain this customer.")
        else:
            st.success(f"✅ **SAFE.** Low probability of churn: **{probability * 100:.1f}%**")