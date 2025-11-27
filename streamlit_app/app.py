import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import shap
import json

from utils import get_mongo_client, log_prediction


# -----------------------------------------------
# CONFIG
# -----------------------------------------------
API_URL = "https://sepsis-api.onrender.com/predict"

MONGO_URI = st.secrets.get("mongo_uri", "")
client = get_mongo_client(MONGO_URI)
db = client["sepsis_logs"]["predictions"]

st.set_page_config(page_title="Sepsis Prediction Dashboard", layout="wide")

st.title("🩺 Sepsis Prediction Dashboard")

st.markdown("""
This dashboard allows:
- Single patient prediction  
- Batch prediction (CSV upload)  
- Analytics & Visualization  
- See past predictions logged in MongoDB  
""")

menu = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Batch Prediction", "Analytics", "History"]
)

# ---------------------------------------------------
# SINGLE PREDICTION
# ---------------------------------------------------
if menu == "Single Prediction":
    st.header("Single Patient Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Hour = st.number_input("Hour", 0, 48, 0)
        HR = st.number_input("HR (Heart Rate)", 0, 220, 90)
        O2Sat = st.number_input("O2Sat %", 0, 100, 95)
        SBP = st.number_input("SBP", 0, 300, 120)
        MAP = st.number_input("MAP", 0, 200, 80)
        DBP = st.number_input("DBP", 0, 200, 70)

    with col2:
        Resp = st.number_input("Resp Rate", 0, 60, 18)
        Age = st.number_input("Age", 0, 110, 45)
        Gender = st.selectbox("Gender", [0, 1])
        Unit1 = st.selectbox("Unit1", [0, 1])
        Unit2 = st.selectbox("Unit2", [0, 1])
        HospAdmTime = st.number_input("HospAdmTime", -500, 500, 0)
        ICULOS = st.number_input("ICULOS", 0, 1000, 5)

    if st.button("Predict"):
        payload = {
            "Hour": Hour,
            "HR": HR,
            "O2Sat": O2Sat,
            "SBP": SBP,
            "MAP": MAP,
            "DBP": DBP,
            "Resp": Resp,
            "Age": Age,
            "Gender": Gender,
            "Unit1": Unit1,
            "Unit2": Unit2,
            "HospAdmTime": HospAdmTime,
            "ICULOS": ICULOS
        }

        response = requests.post(API_URL, json=payload)
        result = response.json()

        prob = result["probability"]
        label = result["predicted_label"]
        threshold = result["threshold_used"]

        # Raw output
        st.success(f"Probability: **{prob:.4f}**")
        st.info(f"Predicted Label: **{label}**")
        st.write(f"Threshold Used: {threshold}")

        # -----------------------------------------
        # HUMAN-READABLE CLINICAL INTERPRETATION
        # -----------------------------------------
        pct = round(prob * 100, 2)

        if label == 1:
            st.error(
                f"🚨 **High Risk of Sepsis Detected**\n\n"
                f"The model predicts a **{pct}% chance the patient has Sepsis.**\n"
                f"Immediate clinical evaluation is recommended."
            )
        else:
            st.success(
                f"🟢 **Low Risk — No Sepsis Detected**\n\n"
                f"The model predicts a **{pct}% probability that the patient is NOT septic.**\n"
                f"Continue monitoring as needed."
            )

        # Log to MongoDB
        log_prediction(db, payload, result, source="streamlit")

# ---------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------
elif menu == "Batch Prediction":
    st.header("Batch Prediction using CSV Upload")

    file = st.file_uploader("Upload CSV with patient vitals", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Uploaded Data", df.head())

        results = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            resp = requests.post(API_URL, json=payload).json()
            results.append(resp)

            # Log each row
            log_prediction(db, payload, resp, source="batch-upload")

        df_results = pd.DataFrame(results)
        st.write("Prediction Results", df_results)

# ---------------------------------------------------
# ANALYTICS
# ---------------------------------------------------
elif menu == "Analytics":
    st.header("Analytics & Visualizations")

    data = list(db.find())
    if len(data) == 0:
        st.warning("No logs found.")
    else:
        df = pd.DataFrame(data)

        st.subheader("Distribution of Predicted Labels")
        fig = px.histogram(df, x="label", nbins=2)
        st.plotly_chart(fig)

        st.subheader("Probability Distribution")
        fig2 = px.histogram(df, x="probability", nbins=20)
        st.plotly_chart(fig2)

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------
elif menu == "History":
    st.header("Past Predictions (MongoDB Logs)")

    data = list(db.find())
    if len(data) == 0:
        st.info("No logs found.")
    else:
        df = pd.DataFrame(data)
        st.write(df[["timestamp", "probability", "label", "source"]])