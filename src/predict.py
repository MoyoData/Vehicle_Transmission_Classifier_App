import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving model
import mlflow
import mlflow.sklearn

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

    def load_processed_data(self):
        """Load the preprocessed data."""
        self.df = pd.read_csv(self.file_path)

        # Identify categorical columns (object dtype)
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Encode categorical features using Label Encoding
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])  # Transform column
            self.label_encoders[col] = le  # Store the encoder for inverse transform if needed
        
        return self.df

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        :param target_column: The name of the target column
        :param test_size: The proportion of the dataset to include in the test split
        :param random_state: Controls the shuffling applied to the data before applying the split
        """
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, model=RandomForestClassifier(n_estimators=100, random_state=42)):
        """
        Train a machine learning model.

        :param model: The machine learning model to train (default is RandomForestClassifier)
        """
        self.model = model
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def save_predictions(self, y_pred, output_path="predictions.csv"):
        """
        Save the predictions to a CSV file.

        :param y_pred: Predicted values
        :param output_path: Path to save the predictions
        """
        predictions_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    def save_model(self, model_path="random_forest_model.pkl"):
        """
        Save the trained model to a file using joblib.

        :param model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def log_mlflow(self, target_column, model_params=None):
        """
        Log model parameters, metrics, and artifacts to MLflow.

        :param target_column: The target column used for training
        :param model_params: Optional dictionary of model parameters to log
        """
        with mlflow.start_run():
            # Log model parameters
            if model_params:
                for param, value in model_params.items():
                    mlflow.log_param(param, value)

            # Log the model
            mlflow.sklearn.log_model(self.model, "random_forest_model")

            # Log metrics like accuracy
            accuracy = self.model.score(self.X_test, self.y_test)
            mlflow.log_metric("accuracy", accuracy)

            # Log the predictions as an artifact
            y_pred = self.predict()
            self.save_predictions(y_pred, 'predictions.csv')
            mlflow.log_artifact("predictions.csv")

            print(f"MLflow run completed with accuracy: {accuracy}")

if __name__ == "__main__":
    # Predict using the processed data
    predictor = DataPredictor('data/processed/processed_data.csv')
    predictor.load_processed_data()

    # Split the data
    predictor.split_data(target_column='transmission_from_vin')  # Assuming 'transmission_from_vin' is the target column

    # Train the model
    predictor.train_model()

    # Log MLflow tracking
    predictor.log_mlflow(target_column='transmission_from_vin', model_params={"n_estimators": 100, "random_state": 42})

    # Save model and predictions
    predictor.save_model("random_forest_model.pkl")
    predictor.save_predictions(predictor.predict(), 'predictions.csv')
