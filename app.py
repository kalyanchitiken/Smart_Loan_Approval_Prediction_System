# ==========================================
# 💰 Smart Loan Approval System (Streamlit App)
# With Sidebar Navigation & EDA + Prediction
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

# --- Sidebar Navigation ---
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Introduction", "📌 Problem Statement", "📊 EDA", "⚙️ Model Building", "🤖 Prediction", "📘 Conclusion"]
)

# ===============================
# 🏠 INTRODUCTION
# ===============================
if page == "🏠 Introduction":
    st.title("💰 Smart Loan Approval Prediction System")
    st.write("""
    Welcome to the **Smart Loan Approval System** — a data-driven machine learning application 
    designed to predict whether a loan application will be **Approved** or **Rejected**.

    This system helps financial institutions make consistent and fair lending decisions 
    using key applicant details such as **income, CIBIL score, and asset values**.
    """)
    st.markdown("---")
    st.subheader("✨ Features")
    st.markdown("""
    - Automated loan approval prediction  
    - Real-time feedback on applicant eligibility  
    - Interactive Exploratory Data Analysis (EDA)  
    - Simple, transparent interface for non-technical users  
    """)

# ===============================
# 📌 PROBLEM STATEMENT
# ===============================
elif page == "📌 Problem Statement":
    st.title("📌 Problem Statement")
    st.write("""
    The manual process of evaluating loan applications is **time-consuming, inconsistent, and prone to bias**.
    
    Financial institutions require an **automated, data-driven system** that can accurately predict loan approval outcomes
    using applicant attributes such as income, dependents, CIBIL score, and asset values.
    
    This project aims to develop a **Random Forest–based ML model** to streamline loan approval decisions 
    and improve transparency in financial risk assessment.
    """)

# ===============================
# 📊 EDA
# ===============================
elif page == "📊 EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    df = pd.read_csv("loan_approval_dataset.csv")

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Automatically detect target column
    target_col = None
    for col in df.columns:
        if "status" in col.lower():
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]

    st.success(f"Detected target column: **{target_col}**")

    # Loan Status Distribution
    st.subheader("🏷️ Loan Status Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=df[target_col], palette="coolwarm", ax=ax)
    ax.set_title("Loan Status Count")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    num_df = df.select_dtypes(include=["int64", "float64"])
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.info("EDA shows that income, CIBIL score, and asset values have strong influence on loan approval likelihood.")

# ===============================
# ⚙️ MODEL BUILDING
# ===============================
elif page == "⚙️ Model Building":
    st.title("⚙️ Model Building Process")
    st.write("""
    The Random Forest algorithm was used to build this loan approval model.
    
    **Steps followed:**
    1. Data cleaning and preprocessing  
    2. Encoding categorical variables (`education`, `self_employed`)  
    3. Splitting data into training and testing sets  
    4. Model training using Random Forest  
    5. Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC AUC  
    
    **Model Performance Summary:**
    - Accuracy: 98%
    - Precision: 98%
    - Recall: 98%
    - ROC-AUC: 0.99
    """)

# ===============================
# 🤖 PREDICTION
# ===============================
elif page == "🤖 Prediction":
    st.title("🤖 Loan Approval Prediction")

    with st.form("prediction_form"):
        st.subheader("Enter Applicant Details:")

        no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
        income_annum = st.number_input("Annual Income (₹)", 100000, 10000000, 500000)
        loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 200000)
        loan_term = st.number_input("Loan Term (Years)", 1, 30, 10)
        cibil_score = st.number_input("CIBIL Score", 300, 900, 750)
        residential_assets_value = st.number_input("Residential Assets Value", 0, 10000000, 500000)
        commercial_assets_value = st.number_input("Commercial Assets Value", 0, 10000000, 100000)
        luxury_assets_value = st.number_input("Luxury Assets Value", 0, 10000000, 100000)
        bank_asset_value = st.number_input("Bank Asset Value", 0, 10000000, 100000)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

        submit_btn = st.form_submit_button("🔍 Predict Loan Status")

    if submit_btn:
        # Prepare input
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

        probs = model.predict_proba(new_data)[0]
        classes = model.named_steps['model'].classes_
        pred_label = model.predict(new_data)[0]

        pred_label_clean = str(pred_label).strip()
        class_names_clean = [str(c).strip() for c in classes]
        pred_index = class_names_clean.index(pred_label_clean)
        pred_proba = probs[pred_index]

        st.subheader("📢 Prediction Result:")
        st.write(f"**Predicted Loan Status:** {pred_label_clean}")
        st.write(f"**Prediction Confidence:** {round(pred_proba * 100, 2)}%")

        if pred_label_clean.lower() == "approved":
            st.success("🎉 Congratulations! Your loan application is likely to be approved. Keep up your financial discipline!")
        else:
            st.error("⚠️ Sorry, your loan application might be rejected. Consider improving your CIBIL score or lowering the loan amount.")

# ===============================
# 📘 CONCLUSION
# ===============================
elif page == "📘 Conclusion":
    st.title("📘 Conclusion")
    st.write("""
    The **Smart Loan Approval System** demonstrates how machine learning can support 
    fair and efficient decision-making in financial institutions.
    
    With an accuracy of **98%**, this model provides reliable predictions that help 
    banks assess loan applications quickly while maintaining transparency.

    **Future Enhancements:**
    - Integration with live CIBIL APIs  
    - Model retraining with new applicant data  
    - Web-based loan management dashboard  
    """)
