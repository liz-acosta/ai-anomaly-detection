import numpy as np
import pandas as pd
from utilities.config import FILEPATHS

# Filepath
DATA_FILE = FILEPATHS["generated_data"]

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

# Main function
def main():
    print("Generating sample e-commerce data .csv file.")

    generate_data()

    print(f".csv file saved to {DATA_FILE}")

# Run the script
if __name__ == "__main__":
    main()