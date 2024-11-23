# About this app
The repo contains an application that performs the following:
* Uses a machine learning model to detect anomalies in e-commerce data
* Uses a large language model to generate a possible explanation for the detected anomalies and recommend actions to take
* Provides a UI that visualizes the e-commerce data and enables users to interact with it
![A gif demo of the app](static/app-demo.gif)

# About this repo
- `data/`: Stores the generated datasets and the model
- `tests/`: Unit tests for the machine learning workflow
- `utilities/`: Contains the scripts for the machine learning workflow as well as any other files used throughout the project

# Prerequisites
* Python 3.8+
* An [OpenAI API key](https://platform.openai.com/docs/api-reference/authentication)

# Set up
1. Clone the repo: `git clone https://github.com/liz-acosta/ai-anomaly-detection.git`
2. Create the virtual environment: `virtualenv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

# Machine learning workflow
1. Generate the sample e-commerce data and output it to a .csv file in the `data/` directory: `python3 -m utilities.generate_sample_data`
2. Train the anomaly model and output it to a .pkl file in the `data/` directory: `python3 -m utilities.train_anomaly_model`
3. Detect anomalies and output them to a .csv file in the `data/` directory: `python3 -m utilities.detect_anomalies`

# Run the application
1. Add your OpenAI API key to the `.template_env` file
2. Rename the file: `mv .template_env .env`
3. Run the Streamlit app: `streamlit run app.py`
4. The app should deploy to `http://localhost:8501/`

# Run the tests (Optional)
1. `python3 -m unittest`