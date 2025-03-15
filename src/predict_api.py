import os
import json
from flask import Flask, request, jsonify, abort
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Model paths
MODEL_PATH_1 = os.path.join(PROJECT_ROOT, "models", "vehicle_transmission_model_62ce78f65aaa4384866e78f4f68763ca.pkl")
MODEL_PATH_2 = os.path.join(PROJECT_ROOT, "models", "vehicle_transmission_model_7064dc1fab0a4b3eb76ce16d162c2eb3.pkl")





@app.route('/Vehicle_Transmission_Classifier_API', methods=['GET'])
def home():
    """Home Endpoint: Description of API and expected JSON format."""
    info = {
        "name": "Vehicle Transmission Classifier API",
        "description": "This API allows making predictions using pre-trained machine learning models to classify the transmission type of vehicles. You can use two models to predict whether a vehicle has an automatic or manual transmission based on various input features.",
        "version": "v1.0",
        "endpoints": {
            "/Vehicle_Transmission_Classifier_API": "Home Page",
            "/health_status": "Health Check",
            "/v1/predict1": "Prediction using Model 1",
            "/v2/predict2": "Prediction using Model 2"
        },
        "input_format": {
            "features": "List of features for prediction  [dealer_type, stock_type, mileage, price, model_year, make, model, certified, fuel_type_from_vin, number_price_changes]",
            "example_request": {
                "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
            }
        },
        "example_response": {
            "success": True,
            "prediction": [0]
        }
    }
    return jsonify(info)

@app.route('/health_status', methods=['GET'])
def health_status():
    """Health Endpoint: Check if the API is up and ready."""
    health = {
        "status": "UP",
        "message": "The Vehicle Transmission Classifier API is available and ready to receive requests."
    }
    return jsonify(health)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
    
    
