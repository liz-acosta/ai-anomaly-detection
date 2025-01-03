import unittest
import os
import pandas as pd

from utilities.train_anomaly_model import preprocess_data, train_model
from utilities.config import FILEPATHS

# Filepaths
DATA_FILE = "tests/test_utils/test_ecommerce_data.csv"
MODEL_FILE = "tests/test_utils/test_model.pkl"

class TestTrainAnomalyModel(unittest.TestCase):
 
    def test_preprocess_data(self):
        """Test for preprocess_date function"""
        
        test_results = preprocess_data(DATA_FILE)

        # Verify the date column was removed
        self.assertNotIn("Date", test_results.columns)

    def test_train_model(self):
        """Test for train_model function"""
        
        # Preprocess the test data
        preprocessed_data = preprocess_data(DATA_FILE)

        train_model(preprocessed_data)

        # Verify the file was created
        self.assertTrue(os.path.exists(MODEL_FILE))
