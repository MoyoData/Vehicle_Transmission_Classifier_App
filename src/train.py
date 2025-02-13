import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess_data(self):
        # Inspect data types
        print(self.df.dtypes)

        # Identify and encode categorical columns
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            print(f"Unique values in {col}: {self.df[col].unique()}")
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col])

    def split_data(self):
        X = self.df.drop(columns=['transmission_from_vin'])
        y = self.df['transmission_from_vin']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        return accuracy

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

if __name__ == "__main__":
    trainer = ModelTrainer('data/processed/processed_data.csv')
    trainer.load_data()
    trainer.preprocess_data()  # Preprocess the data to handle categorical columns
    X_train, X_test, y_train, y_test = trainer.split_data()
    model = trainer.train_model(X_train, y_train)
    accuracy = trainer.evaluate_model(X_test, y_test)
    trainer.save_model('vehicle_transmission_model.pkl')