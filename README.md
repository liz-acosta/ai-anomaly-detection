# AI-Powered Anomaly Detection for E-Commerce Analytics

This repository provides a hands-on example of using unsupervised learning to detect anomalies in e-commerce data. 
The application leverages algorithms like Isolation Forest to identify unusual sales, revenue, and traffic patterns.

## Features
- **Data Generation:** Create realistic e-commerce datasets with embedded anomalies.
- **Model Training:** Train an Isolation Forest model to detect anomalies.
- **Anomaly Detection:** Apply the model to new data and extract anomalies.
- **Visualization:** Visualize anomalies in time-series data.

![A gif demo of the app](static/app-demo.gif)

# About this repo

## File structure
- `data/`: Stores the generated datasets and the model
- `static/`: Static resources used in the project
- `tests/`: Unit tests for the machine learning workflow
- `utilities/`: Contains the scripts for the machine learning workflow as well as any other files used throughout the project

## How to set up and run the app

### Prerequisites
* Python 3.8+
* An [OpenAI API key](https://platform.openai.com/docs/api-reference/authentication)

### Set up
1. Clone the repo: `git clone https://github.com/liz-acosta/ai-anomaly-detection.git`
2. Create the virtual environment: `virtualenv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

### Machine learning workflow
1. Generate the sample e-commerce data and output it to a .csv file in the `data/` directory: `python3 -m utilities.generate_sample_data`
2. Train the anomaly model and output it to a .pkl file in the `data/` directory: `python3 -m utilities.train_anomaly_model`
3. Detect anomalies and output them to a .csv file in the `data/` directory: `python3 -m utilities.detect_anomalies`

### Run the application
1. Add your OpenAI API key to the `.template_env` file
2. Rename the file: `mv .template_env .env`
3. Run the Streamlit app: `streamlit run app.py`
4. The app should deploy to `http://localhost:8501/`

### Run the tests (Optional)
1. `python3 -m unittest`

## Additional resources
* [Build an AI-Powered Anomaly Detection Application for E-Commerce Analytics](https://dev.to/lizzzzz/build-an-ai-powered-anomaly-detection-application-for-e-commerce-analytics-2fj)