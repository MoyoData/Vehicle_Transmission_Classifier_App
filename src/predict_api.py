import os
import json
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Define paths for models (ensure these paths are correct)
MODEL_PATH_1 = os.path.join("models", "vehicle_transmission_model_62ce78f65aaa4384866e78f4f68763ca.pkl")
MODEL_PATH_2 = os.path.join("models", "vehicle_transmission_model_7064dc1fab0a4b3eb76ce16d162c2eb3.pkl")


class DataPredictor:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Load a model from a file."""
        self.model = joblib.load(model_path)

    def predict(self, features):
        """Make predictions using the loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict([features])


@app.route('/Vehicle_Transmission_Classifier_API', methods=['GET'])
def home():
    """Home Endpoint: Description of API and expected JSON format."""
    info = {
        "name": "Vehicle Transmission Classifier API",
        "description": "This API allows making predictions using pre-trained machine learning models to classify the transmission type of vehicles. You can use two models to predict whether a vehicle has an automatic or manual transmission based on various input features.",
        "version": "v1.0",
        "endpoints": {
            "/Vehicle_Transmission_Classifier_API'": "Home Page",
            "/helth_status": "Health Check",
            "/v1/predict1": "Prediction using Model 1",
            "/v2/predict1": "Prediction using Model 2"
        },
        "input_format": {
            "features": "List of features for prediction  [dealer_type, stock_type, mileage, price, model_year, make, model, certified, fuel_type_from_vin, number_price_changes]",
            "example_request": {
                "features": ["I", "Used", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
            }
        },
        "example_response": {
            "success": True,
            "prediction": [1]
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


@app.route('/v1/predict1', methods=['POST'])
def predict_v1():
    """Prediction Endpoint v1: Using Model 1."""
    data = request.get_json()

    if 'features' not in data:
        return jsonify({"error": "Missing required field: features"}), 400

    features = data['features']

    # Initialize predictor and load the model
    predictor = DataPredictor()
    predictor.load_model(MODEL_PATH_1)

    # Make prediction
    try:
        prediction = predictor.predict(features)
        return jsonify({
            "success": True,
            "prediction": prediction.tolist()  # Convert numpy array to list for JSON compatibility
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/v2/predict1', methods=['POST'])
def predict_v2():
    """Prediction Endpoint v2: Using Model 2."""
    data = request.get_json()

    if 'features' not in data:
        return jsonify({"error": "Missing required field: features"}), 400

    features = data['features']

    # Initialize predictor and load the model
    predictor = DataPredictor()
    predictor.load_model(MODEL_PATH_2)

    # Make prediction
    try:
        prediction = predictor.predict(features)
        return jsonify({
            "success": True,
            "prediction": prediction.tolist()  # Convert numpy array to list for JSON compatibility
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
