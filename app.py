# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Churn System", layout="wide")

# ===============================
# SIMPLE LOGIN SYSTEM
# ===============================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "musthafa" and password == "1234":
            st.session_state["login"] = True
        else:
            st.error("Invalid Credentials")


# ===============================
# LOAD MODEL FILES
# ===============================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📊 AI Churn System")
page = st.sidebar.radio("Navigation",
                        ["🏠 Dashboard", "📂 Upload Data", "📊 Dataset", "🤖 Predict"])

# ===============================
# DATA STORAGE
# ===============================
if "data" not in st.session_state:
    st.session_state["data"] = pd.read_csv("telco_churn_7000.csv")

df = st.session_state["data"]

# ===============================
# DASHBOARD
# ===============================
if page == "🏠 Dashboard":

    st.title("📊 Business Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['Churn'].value_counts(normalize=True)[1]*100:.2f}%")
    col3.metric("Avg Charges", f"{df['MonthlyCharges'].mean():.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="tenure", color="Churn",
                            title="Tenure Distribution",
                            template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x="Churn", y="MonthlyCharges",
                      title="Charges vs Churn",
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# ===============================
# UPLOAD DATA
# ===============================
elif page == "📂 Upload Data":

    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["data"] = df
        st.success("Dataset Uploaded Successfully!")

# ===============================
# DATASET VIEW
# ===============================
elif page == "📊 Dataset":

    st.title("📊 Dataset View")
    st.dataframe(df.head(100))

# ===============================
# PREDICTION
# ===============================
elif page == "🤖 Predict":

    st.title("🤖 Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure", 1, 72)
        monthly = st.slider("Monthly Charges", 20, 120)

    with col2:
        total = st.number_input("Total Charges", 20.0, 10000.0)

    # SAFE INPUT
    input_data = np.zeros((1, len(columns)))

    for i, col in enumerate(columns):
        if col == "tenure":
            input_data[0][i] = tenure
        elif col == "MonthlyCharges":
            input_data[0][i] = monthly
        elif col == "TotalCharges":
            input_data[0][i] = total
        else:
            input_data[0][i] = 0

    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Risk: {prob*100:.2f}%")
        else:
            st.success(f"✅ Safe: {(1-prob)*100:.2f}%")