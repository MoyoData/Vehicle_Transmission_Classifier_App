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
label_encoders['dealer_type'].fit(['F', 'I'])  
label_encoders['stock_type'].fit(['NEW', 'USED'])
label_encoders['make'].fit(['Ram', 'GMC', 'Lexus', 'Jeep', 'Honda', 'Chevrolet', 'Ford',
       'Nissan', 'Mercedes-Benz', 'Infiniti', 'Alfa Romeo', 'Hyundai',
       'Kia', 'Mitsubishi', 'Toyota', 'Jaguar', 'Audi', 'Dodge',
       'Volkswagen', 'BMW', 'Mazda', 'Cadillac', 'Tesla', 'Volvo', 'Mini',
       'Subaru', 'Acura', 'Land Rover', 'Buick', 'Lincoln', 'Chrysler',
       'Porsche', 'Maserati', 'Pontiac', 'Fiat', 'Scion', 'Genesis',
       'Saturn', 'Hummer', 'Rivian', 'Polestar', 'Smart', 'Suzuki',
       'Mercury']) 
label_encoders['model'].fit(['2500', 'Yukon', 'RX350h', 'Wrangler', 'Odyssey', 'Silverado 1500',
       'F150', 'Sierra 2500', 'Yukon XL', 'Rogue', 'RX-Series', 'GLC',
       'Grand Cherokee', 'Civic', 'Bronco Sport', 'QX50', 'Stelvio',
       'Kona', 'Sierra 1500', 'Escape', 'Mustang', 'Forte', 'Trailblazer',
       '1500', 'Transit 350', 'CR-V', 'Pathfinder', 'Outlander', 'RX350',
       'Tacoma', 'Soul', 'Tucson', 'F-Pace', 'SQ7', 'Colorado',
       'Transit 250', 'CT200h', 'Murano', 'Journey', 'Silverado 2500',
       'Golf', 'SQ5', 'Pilot', 'Hornet', 'X2', 'CX-30', 'RVR', 'Fusion',
       'Elantra', 'XT5', 'Model Y', 'XC60', 'Hatchback', 'Palisade',
       'Q60', 'Outback', 'Sorento', 'Rav4', 'RDX', 'GR86', 'GLE-Class',
       'Atlas', 'Spark', 'Range Rover Evoque', 'GTI', 'Terrain', 'XC90',
       'Kicks', 'Grand Caravan', 'Accord', 'Malibu', 'Golf R',
       'Blazer EV', 'Camry', 'Corolla', 'Encore', 'Aviator', 'Tiguan',
       '1500 Classic', 'Sierra 3500', 'Equinox', 'NX 300',
       'Wrangler JL 2018.5', 'Expedition Max', 'Atlas Cross Sport',
       'Eclipse Cross', 'Mustang MACH-E', 'Focus', 'Durango', 'EV6',
       'Compass', 'E-Class', 'Taos', 'Q5', 'Equinox EV', 'Crosstrek',
       'Corvette', 'Highlander', 'XT6', 'Gladiator', 'WRX', 'Tundra',
       'Q7', 'Sentra', 'Q8', 'Micra', 'Charger', 'Integra', 'Impreza',
       'Santa Fe', 'CX-5', 'Nautilus', 'Edge', 'Santa Fe Sport', 'BRZ',
       'Seltos', 'Venue', 'Renegade', 'ID.4', 'e-tron GT', 'Lyriq',
       'CLS-Class', 'New Sierra 1500', '300', 'Jetta', '3500', 'F350 S/D',
       'Range Rover Sport', 'Taycan', '4Runner', 'Forester',
       'Range Rover Velar', 'Qashqai', 'Frontier', 'Explorer', 'Ghibli',
       'Sportage', 'Mazda3', 'Passport', 'Trax', 'Suburban', 'X1',
       'Traverse', 'Kona EV', 'Ioniq 6', 'New Silverado 1500', 'Mirage',
       'Camaro', 'Encore GX', 'QX80', 'CX-50', 'Cruze', 'A4', 'EQE SUV',
       'IS-Series', 'Forte5', 'HR-V', 'Silverado 3500', 'C-Class',
       'EcoSport', 'CX-90', 'NV 2500 Van', 'Pacifica', 'Cooper',
       'Silverado EV', 'X5', 'Escalade', 'Acadia', 'Telluride',
       'Avalanche 1500', 'G5', 'ATS Sedan', 'Mazda5', 'Canyon', 'MDX',
       'Q3', 'Sienna', 'Passat', 'Wagoneer', 'X7', 'Ioniq 5', '2-Series',
       'X6', '3-Series', 'UX-Series', 'Range Rover', 'X4', 'Cherokee',
       'F250 S/D', 'GLC-Class', 'X3', 'Ridgeline', 'Envision', 'Model 3',
       'Discovery', 'Envista', 'Grand Wagoneer', 'Grand Cherokee L',
       'Ranger', 'XT4', 'Veloster', 'Santa Fe XL', 'Silverado 1500 LTD',
       '5-Series', 'TLX', '3500 Cab/Chassis', 'Accent', 'Sonata', 'Tahoe',
       '200', 'Express Cargo', 'bZ4X', 'Hummer EV SUT', 'Optima', 'CX-9',
       'Stinger', 'Bronco', 'XC40', 'Titan', '500L', 'Mazda6', 'QX60',
       'Tonale', 'Titan XD', 'Altima', 'EQE', 'F150 Lightning', 'M4',
       'Ioniq', 'XLR', 'Q50', 'Enclave', 'GLB', 'Z', 'Wrangler JK',
       'B-Class', 'Maxima', 'S-Class', 'Versa', 'Legacy',
       'Grand Highlander', 'S3', 'Ram 1500', 'Blazer', 'Elantra GT',
       'Golf Sportwagen', '4-Series', '1500 ProMaster', 'EV9',
       'Accord Crosstour', 'Touareg', 'Santa Cruz', 'Navigator',
       'Countryman', 'Sedona', 'Prius Prime', 'Niro', 'ES250',
       'Challenger', '911', 'A5', 'Rio', 'Savana Cargo', 'Q4 e-tron',
       'Expedition', 'CX-3', 'e-tron', 'GLA', 'Ariya', '7-Series', 'FR-S',
       'QX70', 'S60', 'S6', 'S90', 'Ascent', 'M2', 'G70', 'i4', 'GV70',
       'Land Cruiser', 'Solterra', 'GV80', '500e', 'MKX', 'Maverick',
       'C-HR', 'Clarity', 'GX460', 'Juke', 'i5', 'QX55', 'Sprinter 2500',
       'NV200', 'MKC', 'GR Corolla', 'CLA-Class', 'Corolla Cross',
       'Venza', 'Paceman', 'Carnival', 'M-Class', 'Rondo', 'Giulia',
       'Wagoneer L', 'Transit Connect', 'NX-Series', 'UX250h', 'Patriot',
       '8-Series', 'Sierra 1500 LTD', 'Armada', 'Model X', 'Lancer',
       'Impala', 'Bolt EV', 'Xterra', 'Sonic', 'Hummer EV SUV',
       'XV Crosstrek', 'Grand Wagoneer L', 'Fit', 'Defender', 'TX-Series',
       'E-Transit 350', 'Genesis Sedan', 'RX350L', 'GLB-Class',
       'GLK-Class', 'A-Class', 'C40', 'RX450h', 'iX', 'CC', 'Macan',
       'Transit 150', 'CT4', 'CX-7', 'A7', 'Transit 350HD', 'Regal',
       'QX56', 'ProMaster', 'G80', 'Discovery Sport', 'Mazda2', 'V60',
        'CX-5 2016.5', 'CTS', 'Leaf', 'ILX', 'S7', 'XF', 'F450 S/D', 'EQB',
       'F-Type', 'Caliber', 'M8', 'CLE', 'Corsair', 'SRX', 'Rav4 Prime',
       'Taurus', 'RS 3', 'G3500 Vans', 'Vue', 'A3', '3500 ProMaster',
       'Model S', 'Bolt EUV', 'EQS', 'Prius', 'Yaris', 'GR Supra', 'S5',
       'Grecale', 'Versa Note', 'Sprinter 2500 Cargo', 'XTS', '3 Series',
       '370Z', 'S4', 'Fiesta', 'G-Class', 'TJ', 'i7', 'X-Trail', 'JX35',
       'A6', 'Q8 e-tron', 'City Express', 'Corolla iM', 'EQS SUV',
       'Compass 2017.5', 'Silverado LD 1500', 'Convertible', 'Sequoia',
       'Verano', 'ATS Coupe', 'Flex', 'V90', 'CT5', 'Defender 110',
       'Sprinter 2500 Crew', 'MX5 RF', 'i3', 'ES350', 'SL-Class',
       'E250 Vans', 'Eclipse Spyder', 'RS 5', 'Express', 'Matrix',
       'Cayenne', 'CLA', 'Grand Cherokee WK', 'H3', 'AMG GT', 'Volt',
       'Genesis Coupe', 'RS Q8', 'R8', 'SQ8', 'e-Golf', 'NX 200t',
       '6 Series', 'Quattroporte', 'Dart', 'G6', 'Quest', 'Ram 2500',
       'Clubman', 'ProMaster City', 'GS-Series', 'Crown', 'FX45', 'TL',
       '500X', 'LX600', 'MX5', 'XE', 'GLA-Class', 'GT-R', 'Allante',
       'S-Series', 'Beetle', 'RLX', 'Lucerne', 'GLS-Class', '500', 'R1S',
       'Avenger', 'ZDX', '5 Series', '718 Cayman', 'Metris', '6-Series',
       'Sierra Limited 1500', 'eSprinter 2500', 'MX-30', '911 Carrera 4',
       'CT6', 'Levante', 'XJ', 'GLS', '124 Spider', 'I-Pace',
       'Elantra Touring', 'ES-Series', 'Polestar 2', 'XM', '350Z',
       'Nitro', 'K5', 'CTS Sedan', 'Fortwo', 'MKZ', 'GranTurismo',
       'Cruze Limited', 'MX-5', 'Dakota', 'G90', 'Eclipse', 'Mark LT',
       'A8', 'Z3', 'CMax', 'SQ8 e-tron', 'Ram 3500', 'RS4', 'RS 6',
       'Savana', 'Avalon', 'Element', 'Van', 'Eos', 'RX400h', 'Insight',
       'Boxster', 'XK', 'F350SD', 'NX 300h', 'Grand Prix', '86',
       'CTS Coupe', 'Sprinter 3500 Cargo', 'S/D C/C', 'Uplander',
       'Liberty', 'Aspen', 'Sprinter 3500XD', 'Continental', 'LR2',
       'Solstice', 'Firebird', 'Cooper Countryman', 'Town & Country',
       'QX30', 'Grand Vitara', 'Crossfire', 'S10', 'LS 500', 'A4 Allroad',
       'PT Cruiser', 'Mirai', '718 Boxster', 'Defender 90', 'Routan',
       'Allure', 'Commercial Vans', 'TT', 'Golf AllTrack', 'Prius C',
       'S60 2015.5', 'RS 7', 'G35', 'Cabriolet', 'Torrent', 'Orlando',
       'Mirage G4', 'R1T', '2500 ProMaster', 'EX35', 'Cavalier',
       'Spark EV', '1-Series', 'DTS', 'RC-Series', 'E-Pace', 'A6 Allroad',
       'B9 Tribeca', 'Panamera', 'E150 Vans', 'FX37', 'Z4', 'LX-Series',
       'Yukon XL 1500', 'XL-7', 'ES300h', 'CLK-Class', 'Aveo',
       'New Beetle', 'Cayman', 'GL-Class', 'Q70', 'GV60', 'Avalanche',
       'Cobalt', 'C70', 'RL', 'Eldorado', 'Range Rover L460', '300M',
       'S8', 'TTS', 'G2500 Vans', '1 Series', 'Suburban 1500', 'Rabbit',
       'Prius V', 'Five Hundred', 'M3', 'i-MiEV', 'LX570', 'Sky',
       'SLK-Class', 'Sebring', 'Malibu Limited', 'G37', 'Ram 150',
       'Miata', 'Veracruz', 'Envoy', 'Grand Am', 'E350 Vans', 'RC F',
       'Aztek', 'Explorer Sport Trac', 'Rendezvous', 'RX500h', 'DeVille',
       'IS F', 'NSX-T', 'Pursuit', 'LR4', 'Spectra5', 'H2', 'Arteon',
       'Cougar', 'SX4', 'TT RS', '4C', 'tC', 'NV 3500 Van',
       'Defender 130', 'Vibe', 'Montana', 'EL', 'Sprinter 3500',
       'FJ Cruiser', '911 Carrera 2', 'LaCrosse', '4500', 'MKS', 'X-Type',
       'FX35', 'Swift', 'Monte Carlo', 'Cube', 'ES330', 'IS250',
       'SLC-Class', 'Thunderbird', 'V60 2015.5', 'C/K3500', 'RC 350',
       'F250SD', 'Pickup', 'LC 500h', 'Rio5', 'Protege', 'XC60 2015.5',
       'MKT', 'GX-Series', 'Optra 5', 'SSR', 'SC430', 'LR3', 'IS350',
       'Tribute', 'Magentis', 'Rainier', 'Tercel', 'TSX', 'ATS',
       'Fifth Avenue', 'MC20', 'LS460'])  
label_encoders['certified'].fit(['Yes', 'No'])
label_encoders['fuel_type_from_vin'].fit(['Diesel', 'Gasoline', 'Hybrid', 'PHEV', 'Electric', 'Hydrogen','CNG'])



class DataPredictor:
    def _init_(self):
        self.model = None

    def load_model(self, model_path):
        """Load a model from a file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")

    def preprocess_features(self, features):
        """Preprocess input features: encode categorical features."""
        try:
            # Encode categorical features
            features[0] = label_encoders['dealer_type'].transform([features[0]])[0]  # dealer_type
            features[1] = label_encoders['stock_type'].transform([features[1]])[0]  # stock_type
            features[5] = label_encoders['make'].transform([features[5]])[0]  # make
            features[6] = label_encoders['model'].transform([features[6]])[0]  # model
            features[7] = label_encoders['certified'].transform([features[7]])[0]  # certified
            features[8] = label_encoders['fuel_type_from_vin'].transform([features[8]])[0]  # fuel_type_from_vin
        except ValueError as e:
            logging.error(f"Error encoding categorical features: {e}")
            raise ValueError(f"Invalid categorical value: {e}")

        return features

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

@app.route('/v1/predict1', methods=['POST'])
def predict_v1():
    """Prediction Endpoint v1: Using Model 1."""
    if not request.is_json:
        abort(415, description="Unsupported Media Type. Content-Type must be 'application/json'.")

    data = request.get_json()

    if 'features' not in data:
        return jsonify({"error": "Missing required field: features"}), 400

    features = data['features']

    # Initialize predictor and load the model
    predictor = DataPredictor()
    try:
        predictor.load_model(MODEL_PATH_1)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Preprocess features
    try:
        features = predictor.preprocess_features(features)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

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
    

@app.route('/v2/predict2', methods=['POST'])
def predict_v2():
    """Prediction Endpoint v2: Using Model 2."""
    if not request.is_json:
        abort(415, description="Unsupported Media Type. Content-Type must be 'application/json'.")

    data = request.get_json()

    if 'features' not in data:
        return jsonify({"error": "Missing required field: features"}), 400

    features = data['features']

    # Initialize predictor and load the model
    predictor = DataPredictor()
    try:
        predictor.load_model(MODEL_PATH_2)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Preprocess features
    try:
        features = predictor.preprocess_features(features)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

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
    
    
