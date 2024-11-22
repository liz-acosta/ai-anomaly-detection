import pandas as pd
import joblib
import numpy as np
import os

from utilities.config import FILEPATHS

# File paths
DATA_FILE = FILEPATHS["generated_data"]
MODEL_FILE = FILEPATHS["model"]
OUTPUT_FILE = FILEPATHS["anomalies"]

# Function to load the model
def load_model(file_path):
    """Load the trained anomaly detection model."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}. Please train the model first.")
    
    return joblib.load(file_path)

# Function to preprocess data
def preprocess_data(file_path):
    """Load and preprocess the data for anomaly detection."""
    
    data = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Drop non-numeric columns for predictions
    numeric_data = data.drop(columns=["Date"])
    
    return data, numeric_data

# Function to detect anomalies
def detect_anomalies(model, numeric_data, original_data):
    """Use the trained model to detect anomalies in the dataset."""
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    predictions = model.predict(numeric_data)
    
    # Add anomaly labels to the original data
    original_data["Anomaly"] = predictions
    anomalies = original_data[original_data["Anomaly"] == -1]
    
    return anomalies

# Function to save detected anomalies
def save_anomalies(anomalies, file_path):
    """Save the detected anomalies to a CSV file."""
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    anomalies.to_csv(file_path, index=False)

# Main function
def main():
    # Step 1: Load the model
    print("Loading trained model...")
    model = load_model(MODEL_FILE)

    # Step 2: Load and preprocess the data
    print("Loading and preprocessing data...")
    original_data, numeric_data = preprocess_data(DATA_FILE)

    # Step 3: Detect anomalies
    print("Detecting anomalies...")
    anomalies = detect_anomalies(model, numeric_data, original_data)

    # Step 4: Save the detected anomalies
    print(f"Saving detected anomalies to {OUTPUT_FILE}...")
    save_anomalies(anomalies, OUTPUT_FILE)

    print(f"Detection complete! {len(anomalies)} anomalies detected.")
    print(anomalies)

# Run the script
if __name__ == "__main__":
    main()