import numpy as np
import pandas as pd
from utilities.config import FILEPATHS

DATA_FILE = FILEPATHS["generated_data"]

MESSAGE_BEGIN = ["###", "Generating sales_data.csv file. This data will be used in the tutorial.", ".", "..", "..."]
MESSAGE_END = [f"Data generated and saved to {FILEPATHS['generated_data']}.", "###"]

def display_message(message):
    for line in message:
        print(line)

def generate_data():
    """Generate sample e-commerce data and save it to a csv file."""

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate normal daily data for 30 days
    dates = pd.date_range(start="2024-01-01", periods=30)
    normal_sales = np.random.normal(loc=500, scale=50, size=30)
    normal_revenue = normal_sales * np.random.normal(loc=20, scale=2, size=30)
    normal_traffic = np.random.normal(loc=2000, scale=300, size=30)

    # Add anomalies (e.g., sudden spikes or drops)
    anomalies = pd.DataFrame({
        "Date": ["2024-01-10", "2024-01-20", "2024-01-25"],
        "Sales": [1000, 200, 800],
        "Revenue": [50000, 4000, 16000],
        "Traffic": [5000, 500, 4000]
    })

    # Convert "Date" column in anomalies to datetime
    anomalies["Date"] = pd.to_datetime(anomalies["Date"])

    # Combine normal and anomalous data
    data = pd.DataFrame({
        "Date": dates,
        "Sales": normal_sales,
        "Revenue": normal_revenue,
        "Traffic": normal_traffic
    })
    
    data = pd.concat([data, anomalies]).sort_values(by="Date").reset_index(drop=True)

    # Save dataset to a .csv file
    csv_filename = DATA_FILE
    data.to_csv(csv_filename, index=False)

def main():

    display_message(MESSAGE_BEGIN)

    generate_data()

    display_message(MESSAGE_END)

if __name__ == "__main__":
    main()