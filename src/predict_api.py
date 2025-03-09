
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

app = Flask(__name__)



# Load models 
model_v1 = joblib.load('/home/chiomau/Vehicle_Transmission_Classifier_App-2/models/random_forest_model.pkl')
model_v2 = joblib.load('/home/chiomau/Vehicle_Transmission_Classifier_App-2/models/random_forest_model.pkl')

# Home Endpoint
@app.route('/vehicle_transmission_home')
def home():
    return jsonify({
        "message": "Welcome to the Vehicle Transmission Classifier API",
        "description": "This API predicts the transmission type of a vehicle based on its features like year, make, model, mileage, and price.",
        "v1_predict_example": {
            "year": 2020,
            "make": "Toyota",
            "model": "Camry",
            "mileage": 50000,
            "price": 25000
        },
        "v2_predict_example": {
            "year": 2021,
            "make": "Honda",
            "model": "Civic",
            "mileage": 30000,
            "price": 22000
        }
    })

# Health Endpoint
@app.route('/health_status')
def health_status():
    return jsonify({"status": "API is running and ready!"})

# Predict Endpoint v1
@app.route('/v1/predict1', methods=['POST'])
def predict_v1():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Validate required fields in the payload
        if not all(key in data for key in ["year", "make", "model", "mileage", "price"]):
            return jsonify({"error": "Missing required fields in the payload"}), 400
        
        # Extract features
        year = data["year"]
        make = data["make"]
        model = data["model"]
        mileage = data["mileage"]
        price = data["price"]
        
        # Predict using the model
        prediction = model_v1.predict([[year, make, model, mileage, price]])
        
        # Return the prediction
        return jsonify({"prediction": prediction[0], "message": "Prediction from model v1"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Predict Endpoint v2
@app.route('/v2/predict1', methods=['POST'])
def predict_v2():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Validate required fields in the payload
        if not all(key in data for key in ["year", "make", "model", "mileage", "price"]):
            return jsonify({"error": "Missing required fields in the payload"}), 400
        
        # Extract features
        year = data["year"]
        make = data["make"]
        model = data["model"]
        mileage = data["mileage"]
        price = data["price"]
        
        # Predict using the second model
        prediction = model_v2.predict([[year, make, model, mileage, price]])
        
        # Return the prediction
        return jsonify({"prediction": prediction[0], "message": "Prediction from model v2"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

