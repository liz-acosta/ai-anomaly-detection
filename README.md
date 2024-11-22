# Set up the environment
1. Create a virtual environment: `virtualenv venv`
2. Activate virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

# Add the OpenAI API key
1. Add OpenAI API key to `.env_template`
2. Rename .env file: `mv .env_template .env`
 
# Generate the app resources
1. Generate e-commerce data: `python3 -m utilities.generate_sample_data`
2. Train anomalies model: `python3 -m utilities.train_anomaly_model`
3. Detect anomalies: `python3 -m utilities.detect_anomalies`

# Run the Streamlit app
1. `streamlit run app.py`

# Run the tests (Optional)
1. `python3 -m unittest`
