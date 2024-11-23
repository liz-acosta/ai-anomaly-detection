import unittest
import os
import pandas as pd
from utilities.generate_ecommerce_data import generate_data
from utilities.config import FILEPATHS

# Filepaths
DATA_FILE = "tests/test_utils/test_ecommerce_data.csv"
MODEL_FILE = "tests/test_utils/test_model.pkl"

class TestGenerateSampleData(unittest.TestCase):
 
    def test_generate_data(self):
        """Test for generate_data function"""

        generate_data()
        
        # Verify the file was created
        self.assertTrue(os.path.exists(DATA_FILE))
        
        # Verify the data in the file
        data = pd.read_csv(DATA_FILE)
        
        expected_columns = ["Date", "Sales", "Revenue", "Traffic"]
        self.assertEqual(list(data.columns), expected_columns)

        expected_rows = 33
        self.assertEqual(len(data), expected_rows)

        # Verify the data contains anomalies
        expected_anomalies = data[
        (data["Sales"] > 700) | (data["Sales"] < 300) |
        (data["Revenue"] > 40000) | (data["Traffic"] > 4000)]
        self.assertEqual(len(expected_anomalies), 3)