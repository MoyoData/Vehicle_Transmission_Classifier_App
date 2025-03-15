# Vehicle Transmission Classifier API Documentation

This document provides detailed information about the Vehicle Transmission Classifier API, including its endpoints, expected request formats, possible responses, and usage examples. The API is designed to classify the transmission type of vehicles (automatic or manual) based on various input features using pre-trained machine learning models.

## Overview

The Vehicle Transmission Classifier API is a Flask-based web service that provides two machine learning models for predicting the transmission type of vehicles. The API accepts a set of features related to a vehicle and returns a prediction indicating whether the vehicle has an automatic (0) or manual (1) transmission.

Key Features:
Two pre-trained models for prediction.

Support for categorical and numerical input features.

Detailed error handling and validation.

Health check endpoint for monitoring API status.

## Running the API
Python 3.7 or higher installed on your system.
pip (Python package manager) installed.

Clone the Repository
https://github.com/MoyoData/Vehicle_Transmission_Classifier_App

Create and activate the Virtual Environment

Install Dependencies: pip install -r requirements.txt

Navigate to the Project Directory: Ensure you are in the directory containing the predict_api.py file

Run the Flask application

Verify the Server is Running: The API will be available at http://127.0.0.1:5000


## Endpoints

[URL: /Vehicle_Transmission_Classifier_API](http://127.0.0.1:5000/Vehicle_Transmission_Classifier_API)

Method: GET

Description: Provides a description of the API, its endpoints, and the expected input format.

Response:
{
  "name": "Vehicle Transmission Classifier API",
  "description": "This API allows making predictions using pre-trained machine learning models to classify the transmission type of vehicles...",
  "version": "v1.0",
  "endpoints": {
    "/Vehicle_Transmission_Classifier_API": "Home Page",
    "/health_status": "Health Check",
    "/v1/predict1": "Prediction using Model 1",
    "/v2/predict2": "Prediction using Model 2"
  },
  "input_format": {
    "features": "List of features for prediction [dealer_type, stock_type, mileage, price, model_year, make, model, certified, fuel_type_from_vin, number_price_changes]",
    "example_request": {
      "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
    }
  },
  "example_response": {
    "success": true,
    "prediction": [0]
  }
}

## Health Check Endpoint

[URL: /health_status](http://127.0.0.1:5000/health_status)

Method: GET

Description: Checks if the API is up and running.

Response: {
  "message": "The Vehicle Transmission Classifier API is available and ready to receive requests.",
  "status": "UP"
}


## Prediction Endpoint v1
Method: POST /v1/predict1

Description: Makes a prediction using Model 1.

Request Body:
{
  "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}

Response:
{
  "success": true,
  "prediction": [0]
}

## Prediction Endpoint v2
Method: POST/v2/predict2

Description: Makes a prediction using Model 2.

Request Body:
{
  "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}

Response: 
{
  "success": true,
  "prediction": [1]
}

## Input Format
The API expects a JSON object with the following structure:

Required Fields:
features: A list of 10 features in the following order:

dealer_type: Categorical (F or I).

stock_type: Categorical (New or Used).

mileage: Numeric (e.g., 15000).

price: Numeric (e.g., 23000).

model_year: Numeric (e.g., 2018).

make: Categorical (e.g., Toyota).

model: Categorical (e.g., Corolla).

certified: Categorical (Yes or No).

fuel_type_from_vin: Categorical (Diesel, Gasoline, Hybrid, PHEV, Electric, Hydrogen, CNG).

number_price_changes: Numeric (e.g., 2).

## Example requestfor Home Endpoint
curl http://127.0.0.1:5000/Vehicle_Transmission_Classifier_API

## Example request for Health Check
curl http://127.0.0.1:5000/health_status


## Example Requests and Responses for the predict endpoints
Use curl or Postman to send a POST request with JSON data

Prediction v1:
curl -X POST http://127.0.0.1:5000/v1/predict1 \
-H "Content-Type: application/json" \
-d '{
  "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}'
Prediction v2:

curl -X POST http://127.0.0.1:5000/v2/predict2 \
-H "Content-Type: application/json" \
-d '{
  "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}'
Example 1: Successful Prediction
Request:
{
  "features": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}
Response: 
{
  "success": true,
  "prediction": [0]
}

Example 2: Missing Required Field
Request:
{
  "invalid_key": ["F", "USED", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}
Response: 
{
  "error": "Missing required field: features"
}

Example 3: Invalid Categorical Value
Request:
{
  "features": ["F", "INVALID", 15000, 23000, 2018, "Toyota", "Corolla", "Yes", "Gasoline", 2]
}
Response: 
{
  "success": false,
  "error": "Invalid categorical value: stock_type"
}

## Error Handling
The API provides detailed error messages for common issues:

400 Bad Request: Missing or invalid input fields.

415 Unsupported Media Type: Request content type is not application/json.

500 Internal Server Error: Model file not found or other server-side issues.



