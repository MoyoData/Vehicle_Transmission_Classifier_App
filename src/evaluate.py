import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ModelEvaluator:
    def __init__(self, run_id, file_path):
        self.run_id = run_id
        self.file_path = file_path
        self.df = None
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_processed_data(self):
        """Load the preprocessed data."""
        self.df = pd.read_csv(self.file_path)

        # Identify categorical columns (object dtype)
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Encode categorical features using Label Encoding (same encoding as in training)
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])  # Transform column
        
        return self.df

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        _, self.X_test, _, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def load_model(self):
        """Load the model from MLflow."""
        model_uri = f"runs:/{self.run_id}/model"  # Load the model by the run ID
        self.model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow Run ID: {self.run_id}")

    def evaluate(self):
        """Evaluate the model on the test data."""
        if self.model is None:
            raise ValueError("Model has not been loaded.")
        
        # Predict using the model
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy and generate classification report
        accuracy = accuracy_score(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        # Log the evaluation metrics in MLflow
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(class_report, "classification_report.txt")

            # Optionally, you can log any other metrics or artifacts
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(class_report)

if __name__ == "__main__":
    # Initialize evaluator with MLflow run ID and dataset path
    run_id = "c1bcad60109046e8b0db0e1a643d5da9"  # Replace with your actual run ID from MLflow
    evaluator = ModelEvaluator(run_id=run_id, file_path="/home/moyo/CMPT2500_Vehicle_Transmission_Classifier_Project/data/processed/processed_data.csv")

    # Load and prepare data
    evaluator.load_processed_data()
    evaluator.split_data(target_column='transmission_from_vin')  

    # Load the trained model from MLflow
    evaluator.load_model()

    # Evaluate the model
    evaluator.evaluate()