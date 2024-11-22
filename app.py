import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from openai import OpenAI
from dotenv import load_dotenv
from utilities.config import FILEPATHS

# Load secrets from .env file
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# File paths
DATA_FILE = FILEPATHS["generated_data"]
MODEL_FILE = FILEPATHS["model"]
ANOMALIES_FILE = FILEPATHS["anomalies"]

# Load Data and Model
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    anomalies = pd.read_csv(ANOMALIES_FILE, parse_dates=["Date"])
    return data, anomalies

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    return model

# Explain Anomalies with OpenAI
def explain_anomaly_openai(row):
    """Use OpenAI to summarize the anomaly and suggest actions."""
    # Format the data row as a readable text
    prompt = f"""
    Analyze the following anomaly in e-commerce data and provide a summary along with suggested actions:
    - Date: {row['Date']}
    - Sales: {row['Sales']}
    - Revenue: {row['Revenue']}
    - Traffic: {row['Traffic']}
    
    The anomaly appears to be abnormal compared to typical patterns. Summarize why this might be happening and suggest actions to address or investigate it.
    """
    
    # Call OpenAI API
    openai_client = OpenAI(api_key=OPENAI_KEY)
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a data analyst specialized in e-commerce."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Load Data and Model
data, anomalies = load_data()
model = load_model()

# Streamlit App Layout
st.title("E-Commerce Anomaly Detection Dashboard with AI-Powered Summaries")

st.header("E-Commerce Data Overview")
st.line_chart(data.set_index("Date")[["Sales", "Revenue", "Traffic"]])
st.subheader("Dataset")
st.write(data)

st.header("Detected Anomalies")
st.write(anomalies)

# Visualize anomalies
metric = st.selectbox("Choose a metric to visualize anomalies", ["Sales", "Revenue", "Traffic"])
plt.figure(figsize=(10, 5))
plt.plot(data["Date"], data[metric], label="Normal Data")
plt.scatter(anomalies["Date"], anomalies[metric], color="red", label="Anomalies", zorder=5)
plt.xlabel("Date")
plt.ylabel(metric)
plt.legend()
st.pyplot(plt)
    
st.header("Explain Anomalies with AI")

# Select an anomaly to explain
selected_anomaly = st.selectbox("Choose an anomaly to explain", anomalies["Date"])
anomaly_row = anomalies.loc[anomalies['Date'] == selected_anomaly]

# Explain using OpenAI
if st.button("Generate Explanation and Actions"):
    with st.spinner("Generating explanation..."):
        explanation = explain_anomaly_openai(anomaly_row)
    st.subheader("AI Explanation and Suggested Actions")
    st.write(explanation)