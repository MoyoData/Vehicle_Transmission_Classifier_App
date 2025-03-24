import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving model
import mlflow
import mlflow.sklearn
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure predictions directory exists
os.makedirs("predictions", exist_ok=True)

class DataPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}  # Dictionary to store label encoders
        logging.info("DataPredictor initialized.")

    def load_processed_data(self):
        """Load the preprocessed data."""
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

        # Identify categorical columns (object dtype)
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Encode categorical features using Label Encoding
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])  # Transform column
            self.label_encoders[col] = le  # Store the encoder for inverse transform if needed
        
        logging.info("Data encoding completed.")
        return self.df

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        try:
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logging.info("Data split into training and testing sets.")
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise

    def train_model(self, model=RandomForestClassifier(n_estimators=100, random_state=42)):
        """Train a machine learning model."""
        try:
            self.model = model
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed.")
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise

    def predict(self):
        """Make predictions on the test data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.model.predict(self.X_test)
        logging.info("Predictions made successfully.")
        return y_pred

    def save_predictions(self, y_pred, output_path="predictions/predictions.csv"):
        """Save the predictions to a CSV file."""
        try:
            predictions_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
            predictions_df.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving predictions: {e}")
            raise

    def save_model(self, model_path="random_forest_model.pkl"):
        """Save the trained model to a file using joblib."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            joblib.dump(self.model, model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def log_mlflow(self, target_column, model_params=None):
        """Log model parameters, metrics, and artifacts to MLflow."""
        try:
            with mlflow.start_run():
                if model_params:
                    for param, value in model_params.items():
                        mlflow.log_param(param, value)
                
                input_example = np.array(self.X_train.iloc[0]).reshape(1, -1)
                mlflow.sklearn.log_model(self.model, "random_forest_model", input_example=input_example)
                
                accuracy = self.model.score(self.X_test, self.y_test)
                mlflow.log_metric("accuracy", accuracy)
                
                y_pred = self.predict()
                self.save_predictions(y_pred, 'predictions/predictions.csv')
                mlflow.log_artifact("predictions/predictions.csv")
                
                logging.info(f"MLflow run completed with accuracy: {accuracy}")
        except Exception as e:
            logging.error(f"Error in MLflow logging: {e}")
            raise

if __name__ == "__main__":
    try:
        predictor = DataPredictor('data/processed/processed_data.csv')
        predictor.load_processed_data()
        predictor.split_data(target_column='transmission_from_vin')  
        predictor.train_model()
        predictor.log_mlflow(target_column='transmission_from_vin', model_params={
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'max_features': 'sqrt'
        })
        predictor.save_model("random_forest_model.pkl")
        predictor.save_predictions(predictor.predict(), 'predictions/predictions.csv')
    except Exception as e:
        logging.error(f"Error in script execution: {e}")
