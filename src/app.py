from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/vehicle_transmission_home', methods=['GET'])
def home():
    response = {
        "description": "This API predicts the transmission type (automatic or manual) of a vehicle based on its attributes.",
        "endpoints": {
            "/v1/predict": {
                "method": "POST",
                "description": "Predicts transmission type using a basic model.",
                "expected_json": {
                    "year": 2020,
                    "make": "Toyota",
                    "model": "Corolla",
                    "mileage": 30000,
                    "price": 20000
                }
            },
            "/v2/predict": {
                "method": "POST",
                "description": "Predicts transmission type using an improved model with additional features.",
                "expected_json": {
                    "year": 2020,
                    "make": "Toyota",
                    "model": "Corolla",
                    "mileage": 30000,
                    "price": 20000,
                    "engine_size": 1.8,
                    "fuel_type": "Gasoline"
                }
            }
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
