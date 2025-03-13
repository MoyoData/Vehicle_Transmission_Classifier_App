
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  
import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os


app = Flask(__name__)

# Dynamically get the absolute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Construct absolute paths for models
model_v1_path = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
model_v2_path = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")

# Load models
model_v1 = joblib.load(model_v1_path)
model_v2 = joblib.load(model_v2_path)


@app.route("/Vehicle_Transmission_Classifier", methods = ['GET'])
def home():
    return jsonify({
        'message': 'Welcome to Vehicle transmission classifier ML Prediction API!',
        'endpoints': {
            '/v1/predict': 'Predict using model v1',
            '/v2/predict': 'Predict using model v2',
            '/health_status': 'Check API health status',
            '/Vehicle_Transmission_Classifier_home': 'Information about how to use the API'
        }
    })

@app.route('/health_status', methods = ['GET'])
def health_status():
    return jsonify({'status': 'API is live and running'})

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    data = request.get_json()
    
    # Check if data is valid
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Assuming the data matches the features the model expects
        input_features = np.array([data['features']])
        prediction = model_v1.predict(input_features)
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 400

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    data = request.get_json()
    
    # Check if data is valid
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Assuming the data matches the features the model expects
        input_features = np.array([data['features']])
        prediction = model_v2.predict(input_features)
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 400

if __name__ == "__main__":
    app.run(debug=True)