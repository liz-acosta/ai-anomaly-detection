import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from utilities.config import FILEPATHS

# Filepaths
DATA_FILE = FILEPATHS["generated_data"]
MODEL_FILE = FILEPATHS["model"]

def preprocess_data(file_path):
    """Load and preprocess the data for training the anomaly detection model."""
    data = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Drop the "Date" column because it is not needed for detecting anomalies
    numeric_data = data.drop(columns=["Date"])
    return numeric_data

def train_model(data):
    """Train an Isolation Forest model for anomaly detection."""
    
    # Isolation Forest parameters
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=0.07,  # Approx. % of anomalies expected
        random_state=42
    )
    model.fit(data)
    return model

# Main function
def main():
    print("Model training initiated ...")

    # Step 1: Load and preprocess the data
    print("Loading and preprocessing data...")
    data = preprocess_data(DATA_FILE)

    # Step 2: Train the model
    print("Training the Isolation Forest model...")
    model = train_model(data)

    # Step 3: Save the trained model
    print(f"Saving the model to {MODEL_FILE}...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("Model training complete!")
    print(f"Model .pkl file saved to {DATA_FILE}")

# Run the script
if __name__ == "__main__":
    main()
