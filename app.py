# app.py

import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the model and scaler
# -------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Title and description
# -------------------------------
st.title("🏦 Loan Approval Prediction System")
st.markdown("""
This application predicts whether a **loan application will be Approved or Rejected**.
Fill in the applicant information in the sidebar and click **Predict Loan Status**.
""")

# -------------------------------
# Sidebar for input
# -------------------------------
st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.number_input("Number of Dependents", 0, 10)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income (₹)", 0, 10000000, step=50000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 0, 10000000, step=50000)
loan_term = st.sidebar.number_input("Loan Term (Months)", 0, 480, step=12)
cibil_score = st.sidebar.number_input("CIBIL Score", 300, 900, step=1)
residential_assets_value = st.sidebar.number_input("Residential Assets Value (₹)", 0, 10000000, step=50000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value (₹)", 0, 10000000, step=50000)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value (₹)", 0, 10000000, step=50000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value (₹)", 0, 10000000, step=50000)

# -------------------------------
# Encoding categorical variables
# -------------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------------------------------
# Prepare input array
# -------------------------------
input_data = np.array([[no_of_dependents, education, self_employed,
                        income_annum, loan_amount, loan_term,
                        cibil_score, residential_assets_value,
                        commercial_assets_value, luxury_assets_value,
                        bank_asset_value]])

# Scale input
input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_scaled)[0]

    # Display results
    if prediction == 1:
        st.success("✅ Congratulations! The loan is likely to be **Approved**")
        st.balloons()  # Fun animation 🎈
    else:
        st.error("❌ Sorry! The loan is likely to be **Rejected**")
        st.warning("Consider improving your CIBIL score or financial assets.")

# -------------------------------
# Optional: show input summary
# -------------------------------
st.subheader("Applicant Information Summary")
st.table({
    "Feature": ["Dependents", "Education", "Self Employed", "Income (₹)", "Loan Amount (₹)", "Loan Term (Months)",
                "CIBIL Score", "Residential Assets (₹)", "Commercial Assets (₹)", "Luxury Assets (₹)",
                "Bank Assets (₹)"],
    "Value": [no_of_dependents, "Graduate" if education else "Not Graduate",
              "Yes" if self_employed else "No", income_annum, loan_amount, loan_term,
              cibil_score, residential_assets_value, commercial_assets_value,
              luxury_assets_value, bank_asset_value]
})