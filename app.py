import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.compose._column_transformer as ct

# ---- FIX for sklearn _RemainderColsList error ----
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList
# --------------------------------------------------

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------
st.sidebar.title("🔍 Navigation")
options = ["Introduction", "Problem Statement", "EDA", "Model Building", "Prediction"]
choice = st.sidebar.radio("Go to:", options)

# ----------------------------------
# 1️⃣ INTRODUCTION
# ----------------------------------
if choice == "Introduction":
    st.title("🏦 Loan Approval Prediction using Machine Learning")
    st.markdown("""
    This project predicts whether a loan application will be **Approved** or **Rejected** using 
    a trained **Random Forest Classifier**.

    ### 🎯 Objective
    To help banks and financial institutions make **faster and more accurate** loan decisions 
    based on applicant financial and credit information.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/4221/4221426.png", width=200)

# ----------------------------------
# 2️⃣ PROBLEM STATEMENT
# ----------------------------------
elif choice == "Problem Statement":
    st.header("📄 Problem Statement")
    st.markdown("""
    Financial institutions face challenges in evaluating loan applications quickly and reliably.  
    Manual verification can be time-consuming and error-prone.  

    **Goal:**  
    Build a machine learning model that predicts if a loan should be approved or rejected 
    based on features like income, CIBIL score, and asset values.
    """)

# ----------------------------------
# 3️⃣ EDA
# ----------------------------------
elif choice == "EDA":
    st.header("📊 Exploratory Data Analysis")

    df = pd.read_csv("loan_approval_dataset.csv")

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Dataset Overview")
    st.write(df.describe())

    st.subheader("Loan Approval Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='loan_status', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("CIBIL Score vs Loan Approval")
    fig, ax = plt.subplots()
    sns.boxplot(x='loan_status', y='cibil_score', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

# ----------------------------------
# 4️⃣ MODEL BUILDING
# ----------------------------------
elif choice == "Model Building":
    st.header("🤖 Model Building Overview")
    st.markdown("""
    - **Algorithm Used:** Random Forest Classifier  
    - **Accuracy:** ~98%  
    - **Evaluation Metrics:** Precision, Recall, F1-Score, ROC AUC  

    The model was trained on loan applicant data with preprocessing for both categorical and numerical features.  
    It effectively distinguishes between *Approved* and *Rejected* applications.
    """)

    st.success("✅ Model successfully trained and saved as `loan_model.pkl`")

# ----------------------------------
# 5️⃣ PREDICTION
# ----------------------------------
elif choice == "Prediction":
    st.header("🔮 Loan Approval Prediction")

    st.markdown("Enter applicant details below to predict loan approval:")

    # Input fields
    no_of_dependents = st.number_input("Number of Dependents", 0, 10, 1)
    income_annum = st.number_input("Annual Income (₹)", 0, 10000000, 500000)
    loan_amount = st.number_input("Loan Amount (₹)", 0, 2000000, 300000)
    loan_term = st.number_input("Loan Term (Years)", 1, 30, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 700)
    residential_assets_value = st.number_input("Residential Asset Value", 0, 5000000, 200000)
    commercial_assets_value = st.number_input("Commercial Asset Value", 0, 5000000, 50000)
    luxury_assets_value = st.number_input("Luxury Asset Value", 0, 5000000, 10000)
    bank_asset_value = st.number_input("Bank Asset Value", 0, 5000000, 100000)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    if st.button("🔍 Predict Loan Status"):
        new_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "education": education,
            "self_employed": self_employed
        }])

        # Predict
        probs = model.predict_proba(new_data)[0]
        classes = model.named_steps['model'].classes_
        pred_label = model.predict(new_data)[0]
        pred_proba = probs[list(classes).index(pred_label)]

        st.subheader("📢 Prediction Result:")
        st.write(f"**Predicted Loan Status:** {pred_label}")
        st.write(f"**Prediction Confidence:** {round(pred_proba*100, 2)}%")

        if pred_label == "Approved":
            st.success("🎉 Congratulations! Your loan application is likely to be approved.")
        else:
            st.error("⚠️ Sorry, your loan application might be rejected. "
                     "Consider improving your credit score or reducing the requested loan amount.")

        st.markdown("""
        ---
        **Conclusion:**  
        The Random Forest model provides an accurate and transparent decision-support system 
        for loan evaluations, helping financial institutions make data-driven lending choices efficiently.
        """)
