import unittest
import os
import pandas as pd

from utilities.detect_anomalies import load_model, preprocess_data, detect_anomalies, save_anomalies
from utilities.config import FILEPATHS

# Filepaths
DATA_FILE = "tests/test_utils/test_ecommerce_data.csv"
MODEL_FILE = "tests/test_utils/test_model.pkl"
ANOMALIES_FILE = "tests/test_utils/test_detected_anomalies.csv"

class TestDetectAnomalies(unittest.TestCase):
    
    def test_load_model(self):
        """Test for load_model function"""

        test_results = load_model(MODEL_FILE)

        # Verify the file was loaded with correct parameters
        self.assertEqual(test_results.contamination, 0.07)
        self.assertEqual(test_results.random_state, 42)

    def test_preprocess_data(self):
        """Test for preprocess_date function"""
        
        test_results = preprocess_data(DATA_FILE)

        # Verify the data was preprocessed by parsing the date column to datetime
        # And then removing the Date column  
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_results[0]["Date"]))
        self.assertNotIn("Date", test_results[1].columns)

    def test_detect_anomalies(self):
        """Test for detect_anomalies function"""

        # Load the test model
        test_model = load_model(MODEL_FILE)
        # Preprocess the test data
        test_original_data, test_numeric_data = preprocess_data(DATA_FILE)

        test_results = detect_anomalies(test_model,test_numeric_data, test_original_data)

        # Verify the correct number of anomalies were detected
        self.assertEqual(len(test_results), 3)

    def test_save_anomalies(self):
        """Test for save_anomalies function"""
        # Load the test model
        test_model = load_model(MODEL_FILE)
        # Preprocess the test data
        test_original_data, test_numeric_data = preprocess_data(DATA_FILE)

        test_anomalies = detect_anomalies(test_model,test_numeric_data, test_original_data)

        test_results = save_anomalies(test_anomalies, ANOMALIES_FILE)

        # Verify the file was created
        self.assertTrue(os.path.exists(ANOMALIES_FILE))

        # Clean up after the tests by removing the generated file
        os.remove(ANOMALIES_FILE)