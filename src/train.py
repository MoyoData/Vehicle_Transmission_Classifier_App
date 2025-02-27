import os
import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        """Loads dataset from CSV file."""
        if not os.path.exists(self.data_path):
            logging.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logging.info("Data successfully loaded.")
        return self.df

    def preprocess_data(self):
        """Handles missing values and encodes categorical features."""
        if self.df is None:
            logging.error("Data not loaded. Call load_data() first.")
            return

        logging.info("Preprocessing data...")

        # Check for missing values
        if self.df.isnull().sum().any():
            logging.warning("Missing values detected. Filling missing values.")
            self.df.fillna(self.df.mode().iloc[0], inplace=True)  # Fills with mode for categorical features

        # Identify categorical columns and encode them
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            logging.info(f"Encoding {col} (Unique values: {self.df[col].nunique()})")
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col])

        logging.info("Data preprocessing complete.")

    def split_data(self):
        """Splits dataset into training and test sets."""
        if 'transmission_from_vin' not in self.df.columns:
            logging.error("'transmission_from_vin' column is missing in the dataset.")
            raise ValueError("'transmission_from_vin' column is missing in the dataset.")

        X = self.df.drop(columns=['transmission_from_vin'])
        y = self.df['transmission_from_vin']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.info(f"Data split into train and test sets. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, param_grid):
        """Performs hyperparameter tuning using GridSearchCV."""
        logging.info("Starting hyperparameter tuning with GridSearchCV...")

        # Define the model (RandomForestClassifier)
        model = RandomForestClassifier(random_state=42)

        # GridSearchCV with the correct parameter grid
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
        logging.info(f"Best model found: {best_model}")

        # Set the best model to self.model
        self.model = best_model

        return best_model

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model and returns accuracy."""
        if not self.model:
            logging.error("Model not trained. Call train_model() first.")
            return None

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy:.4f}")

        return accuracy, y_pred

    def save_model(self, model_path):
        """Saves the trained model to a file."""
        joblib.dump(self.model, model_path)
        logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))  # Get the current script's directory
    config_path = os.path.join(project_root, "../configs/train_config.yaml")
    data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")

    # Load training configuration from YAML file
    config = None
    training_compute = "default"  # Default value if not provided

    # Load configuration from YAML
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.info(f"Loaded config: {config}")
                training_compute = config.get("training_compute", "default")
        except Exception as e:
            logging.error(f"Error loading YAML file: {e}")
    else:
        logging.warning(f"Config file not found: {config_path}. Using default settings.")

    # Use MLflow autologging for sklearn
    mlflow.sklearn.autolog()

    # Start an MLflow run to track the experiment
    run_name = f"classification_{config.get('model_params', {}).get('n_estimators', 'default')}_{training_compute}"
    with mlflow.start_run(run_name=run_name) as run:

        logging.info(f"Starting MLflow run: {run.info.run_id} with run name: {run_name}")

        # Dynamically generate model_save_path using the run ID inside the MLflow context
        model_save_path = os.path.join(project_root, f"vehicle_transmission_model_{run.info.run_id}.pkl")

        if config:
            mlflow.log_params(config)  # Logs training parameters

        # Initialize and run training pipeline
        trainer = ModelTrainer(data_path)
        trainer.load_data()
        trainer.preprocess_data()
        X_train, X_test, y_train, y_test = trainer.split_data()

        # Train model with hyperparameter tuning
        param_grid = config.get("model_params", {
            'n_estimators': [100, 150, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False],
            'max_features': ['log2', 'sqrt', None]
        })
        
        model = trainer.train_model(X_train, y_train, param_grid)


        # Create an input example using a single row from the training data
        input_example = X_train.iloc[:1]

        # Log the trained model in MLflow with input example
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

        # Save the model locally
        trainer.save_model(model_save_path)

        logging.info(f"MLflow Run ID: {run.info.run_id}")
