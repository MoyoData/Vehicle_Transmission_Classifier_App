import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

class ModelEvaluator:
    def __init__(self, file_path, model_path):
        self.model_path = model_path
        self.file_path = file_path
        self.model = None
        self.df = None
        self.y_true = None
        self.y_pred = None
        self.label_encoders = {}  # Dictionary to store label encoders

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

        # Identify categorical columns (object dtype)
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        
        # Encode categorical features using Label Encoding
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])  # Transform column
            self.label_encoders[col] = le  # Store the encoder for inverse transform if needed
        
        return self.df

    def evaluate_model(self):
        X = self.df.drop(columns=['transmission_from_vin'])
        self.y_true = self.df['transmission_from_vin']
        self.y_pred = self.model.predict(X)

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self, class_names=None, save_path=None):
        cm = self.get_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_classification_report(self):
        return classification_report(self.y_true, self.y_pred, output_dict=True)

    def save_classification_report_as_csv(self, report, save_path):
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(save_path, index=True)

    def evaluate(self, class_names=None, confusion_matrix_save_path=None, classification_report_save_path=None):
        self.load_model()
        self.load_data()
        self.evaluate_model()
        print("Confusion Matrix:\n", self.get_confusion_matrix())
        report = self.get_classification_report()
        print("Classification Report:\n", pd.DataFrame(report).transpose())
        self.plot_confusion_matrix(class_names, confusion_matrix_save_path)
        if classification_report_save_path:
            self.save_classification_report_as_csv(report, classification_report_save_path)

if __name__ == "__main__":
    evaluator = ModelEvaluator('processed_data.csv', 'random_forest_model.pkl')
    class_names = ['Automatic', 'Manual']
    evaluator.evaluate(
        class_names,
        confusion_matrix_save_path='confusion_matrix.png',
        classification_report_save_path='classification_report.csv'
    )
