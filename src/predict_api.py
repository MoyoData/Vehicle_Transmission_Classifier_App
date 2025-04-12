import os
import json
import time
from flask import Flask, request, jsonify, abort
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import threading
import psutil

# Create the Flask app instance
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app, path='/metrics')

# Custom metrics
prediction_requests = Counter(
    'model_prediction_requests_total', 
    'Total number of prediction requests', 
    ['model_version', 'endpoint', 'status']
)
prediction_time = Histogram(
    'model_prediction_duration_seconds', 
    'Time spent processing prediction', 
    ['model_version', 'endpoint']
)
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')
model_load_time = Histogram(
    'model_load_duration_seconds',
    'Time spent loading models',
    ['model_version']
)

# Configure logging
def configure_logging(log_directory='logs'):
    os.makedirs(log_directory, exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]  # Log to stdout for Docker
    )
    modules = ['data_processing', 'train', 'predict', 'predict_api']
    loggers = {}
    for module in modules:
        logger = logging.getLogger(f'vehicle_classifier.{module}')
        file_handler = logging.FileHandler(f'{log_directory}/{module}.log')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        loggers[module] = logger
    return loggers

# Configure logging
loggers = configure_logging()
predict_api_logger = loggers['predict_api']

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Model paths
MODEL_PATH_1 = os.path.join(PROJECT_ROOT, "models", "vehicle_transmission_model_62ce78f65aaa4384866e78f4f68763ca.pkl")
MODEL_PATH_2 = os.path.join(PROJECT_ROOT, "models", "vehicle_transmission_model_7064dc1fab0a4b3eb76ce16d162c2eb3.pkl")

# Initialize label encoders for categorical features
label_encoders = {
    'dealer_type': LabelEncoder(),
    'stock_type': LabelEncoder(),
    'make': LabelEncoder(),
    'model': LabelEncoder(),
    'certified': LabelEncoder(),
    'fuel_type_from_vin': LabelEncoder()
}

# Fit the label encoders with example data
def initialize_label_encoders():
    label_encoders['dealer_type'].fit(['F', 'I'])
    label_encoders['stock_type'].fit(['NEW', 'USED'])
    label_encoders['make'].fit(['Ram', 'GMC', 'Lexus', 'Jeep', 'Honda', 'Chevrolet', 'Ford', 'Nissan', 'Mercedes-Benz'])
    label_encoders['model'].fit(['2500', 'Yukon', 'RX350h', 'Wrangler', 'Odyssey', 'Silverado 1500', 'F150'])
    label_encoders['certified'].fit(['Yes', 'No'])
    label_encoders['fuel_type_from_vin'].fit(['Diesel', 'Gasoline', 'Hybrid', 'PHEV', 'Electric', 'Hydrogen', 'CNG'])

initialize_label_encoders()

class DataPredictor:
    def __init__(self):
        self.models = {}

    def load_model(self, model_path):
        if model_path in self.models:
            self.model = self.models[model_path]
        else:
            start_time = time.time()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            self.models[model_path] = self.model
            load_time = time.time() - start_time
            model_version = "1.0" if "62ce78f65aaa4384866e78f4f68763ca" in model_path else "2.0"
            model_load_time.labels(model_version=model_version).observe(load_time)
            predict_api_logger.info(f"Model loaded from {model_path} in {load_time:.4f} seconds")
    
    def preprocess_features(self, features):
        try:
            features[0] = label_encoders['dealer_type'].transform([features[0]])[0]
            features[1] = label_encoders['stock_type'].transform([features[1]])[0]
            features[5] = label_encoders['make'].transform([features[5]])[0]
            features[6] = label_encoders['model'].transform([features[6]])[0]
            features[7] = label_encoders['certified'].transform([features[7]])[0]
            features[8] = label_encoders['fuel_type_from_vin'].transform([features[8]])[0]
            return features
        except ValueError as e:
            predict_api_logger.error(f"Error encoding features: {e}")
            raise

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict([features])

def monitor_resources():
    while True:
        try:
            process = psutil.Process(os.getpid())
            memory_usage.set(process.memory_info().rss)
            cpu_usage.set(process.cpu_percent(interval=1))
        except Exception as e:
            predict_api_logger.error(f"Resource monitoring error: {e}")
        time.sleep(15)

@app.route('/metrics')
def metrics_endpoint():
    return metrics.export()

@app.route('/Vehicle_Transmission_Classifier_API', methods=['GET'])
def home():
    return jsonify({
        "name": "Vehicle Transmission Classifier API",
        "version": "v1.0",
        "endpoints": {
            "/Vehicle_Transmission_Classifier_API": "Home Page",
            "/health_status": "Health Check",
            "/v1/predict1": "Prediction using Model 1",
            "/v2/predict2": "Prediction using Model 2",
            "/metrics": "Prometheus Metrics"
        }
    })

@app.route('/health_status', methods=['GET'])
def health_status():
    return jsonify({"status": "UP", "message": "API is ready"})

@app.route('/v1/predict1', methods=['POST'])
def predict_v1():
    return handle_prediction(MODEL_PATH_1, "1.0", "v1/predict1")

@app.route('/v2/predict2', methods=['POST'])
def predict_v2():
    return handle_prediction(MODEL_PATH_2, "2.0", "v2/predict2")

def handle_prediction(model_path, model_version, endpoint):
    start_time = time.time()
    predict_api_logger.info(f"Accessed {endpoint} endpoint")
    try:
        if not request.is_json:
            abort(415, description="Content-Type must be 'application/json'")
        data = request.get_json()
        if 'features' not in data:
            prediction_requests.labels(model_version=model_version, endpoint=endpoint, status="error").inc()
            return jsonify({"error": "Missing features"}), 400
        predictor = DataPredictor()
        predictor.load_model(model_path)
        features = predictor.preprocess_features(data['features'])
        prediction = predictor.predict(features)
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version, endpoint=endpoint).observe(duration)
        prediction_requests.labels(model_version=model_version, endpoint=endpoint, status="success").inc()
        return jsonify({"success": True, "prediction": prediction.tolist()})
    except Exception as e:
        predict_api_logger.error(f"Error in {endpoint}: {e}")
        prediction_requests.labels(model_version=model_version, endpoint=endpoint, status="error").inc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    predict_api_logger.info("Starting application")
    threading.Thread(target=monitor_resources, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)