import unittest
import os
import pandas as pd

from utilities.detect_anomalies import load_model, preprocess_data, detect_anomalies, save_anomalies
from utilities.config import FILEPATHS

DATA_FILE = "tests/test_utils/test_ecommerce_data.csv"
MODEL_FILE = "tests/test_utils/test_model.pkl"
ANOMALIES_FILE = "tests/test_utils/test_detected_anomalies.csv"

class TestDetectAnomalies(unittest.TestCase):
 
    def test_load_model(self):

        test_results = load_model(MODEL_FILE)

        # Verify the file was created
        self.assertEqual(test_results.contamination, 0.07)
        self.assertEqual(test_results.random_state, 42)

    def test_preprocess_data(self):

        test_results = preprocess_data(DATA_FILE)

        # Verify the file was created
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_results[0]["Date"]))
        self.assertNotIn("Date", test_results[1].columns)


    def test_detect_anomalies(self):

        test_model = load_model(MODEL_FILE)
        test_original_data, test_numeric_data = preprocess_data(DATA_FILE)

        test_results = detect_anomalies(test_model,test_numeric_data, test_original_data)

        # Verify the file was created
        self.assertEqual(len(test_results), 3)

    def test_save_anomalies(self):

        test_model = load_model(MODEL_FILE)
        test_original_data, test_numeric_data = preprocess_data(DATA_FILE)

        test_anomalies = detect_anomalies(test_model,test_numeric_data, test_original_data)

        test_results = save_anomalies(test_anomalies, ANOMALIES_FILE)

        # Verify the file was created
        self.assertTrue(os.path.exists(ANOMALIES_FILE))

        # Clean up after the tests by removing `test_detected_anomalies.csv`
        os.remove(ANOMALIES_FILE)