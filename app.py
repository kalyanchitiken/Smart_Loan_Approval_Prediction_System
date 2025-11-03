# ==========================================
# 🧠 Smart Loan Approval Prediction App
# Streamlit + Random Forest + Auto EDA
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.compose._column_transformer as ct

# --- Patch for sklearn _RemainderColsList error ---
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# --- Load trained model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Page setup ---
st.set_page_config(page_title="Smart Loan Approval System", page_icon="💰", layout="wide")
st.title("💰 Smart Loan Approval Prediction System")
st.write("This app uses a trained Random Forest ML model to predict loan approval decisions based on applicant financial details.")

# ===============================
# 📊 Exploratory Data Analysis (EDA)
# ===============================

st.header("🔍 Exploratory Data Analysis (EDA)")

# Load dataset directly from uploaded file (same as used for training)
df = pd.read_csv("loan_approval_dataset.csv")

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# Automatically detect the target column (case-insensitive)
target_col = None
for col in df.columns:
    if 'status' in col.lower():
        target_col = col
        break

if target_col:
    st.success(f"Detected target column: **{target_col}**")
else:
    st.error("Could not detect a target column automatically. Please check your dataset.")
    target_col = df.columns[-1]  # fallback

# --- Loan status distribution plot ---
st.subheader("🏷️ Loan Status Distribution")
fig, ax = plt.subplots(figsize=(5,3))
sns.countplot(x=df[target_col], palette="coolwarm", ax=ax)
ax.set_title("Loan Status Count")
st.pyplot(fig)

# --- Correlation heatmap ---
st.subheader("📈 Correlation Heatmap")
num_df = df.select_dtypes(include=["int64", "float64"])
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.info("EDA shows that applicant income, CIBIL score, and asset values have strong influence on loan approval likelihood.")

# ===============================
# 🤖 Prediction Section
# ===============================
st.header("🤖 Loan Approval Prediction")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Applicant Details:")

    no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
    income_annum = st.number_input("Annual Income (₹)", 100000, 10000000, 500000)
    loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 200000)
    loan_term = st.number_input("Loan Term (in years)", 1, 30, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 750)
    residential_assets_value = st.number_input("Residential Assets Value", 0, 10000000, 500000)
    commercial_assets_value = st.number_input("Commercial Assets Value", 0, 10000000, 100000)
    luxury_assets_value = st.number_input("Luxury Assets Value", 0, 10000000, 100000)
    bank_asset_value = st.number_input("Bank Asset Value", 0, 10000000, 100000)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    submit_btn = st.form_submit_button("🔍 Predict Loan Status")

if submit_btn:
    # Prepare input data
    new_data = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'education': education,
        'self_employed': self_employed
    }])

    # Predict
    probs = model.predict_proba(new_data)[0]
    classes = model.named_steps['model'].classes_
    pred_label = model.predict(new_data)[0]

    # Normalize label
    pred_label_clean = str(pred_label).strip()
    class_names_clean = [str(c).strip() for c in classes]
    pred_index = class_names_clean.index(pred_label_clean)
    pred_proba = probs[pred_index]

    # --- Display results ---
    st.subheader("📢 Prediction Result:")
    st.write(f"**Predicted Loan Status:** {pred_label_clean}")
    st.write(f"**Prediction Confidence:** {round(pred_proba * 100, 2)}%")

    if pred_label_clean.lower() == "approved":
        st.success("🎉 Congratulations! Your loan application is likely to be approved. Maintain good credit behavior to ensure smooth processing.")
    else:
        st.error("⚠️ Sorry, your loan application might be rejected. Consider improving your credit score or reducing the requested loan amount.")

    # --- Conclusion ---
    st.markdown("---")
    st.subheader("📘 Conclusion:")
    st.write(
        "The Random Forest model provides an accurate and transparent decision-support system for loan evaluations, "
        "helping financial institutions make data-driven lending choices efficiently."
    )
