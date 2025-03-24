import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from logging_config import configure_logging

# Configure logging
loggers = configure_logging()
logger = loggers['data_processing']

class DataPreprocessor:
    def __init__(self, file_path, logger):
        self.file_path = file_path
        self.df = None
        self.encoders = {}
        self.logger = logger  # Use the provided logger

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, on_bad_lines='skip')
            self.logger.info(f"Successfully loaded data from {self.file_path}. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            self.logger.error("CSV file is empty")
            return None

    def explore_data(self):
        self.logger.info("Exploring data")
        print(self.df.head())
        print(self.df.tail())
        print(self.df.columns)
        print(self.df.info())
        print(self.df.shape)
        print(self.df.dtypes)
        print(self.df.nunique())
        print(self.df.duplicated().sum())

    def preprocess_data(self):
        try:
            self.logger.info("Starting data preprocessing")

            cols_to_drop = ['dealer_email', 'listing_id', 'listing_heading', 'listing_type', 
                            'listing_url', 'listing_first_date', 'days_on_market', 'dealer_id', 
                            'dealer_name', 'dealer_street', 'dealer_city', 'dealer_province', 
                            'dealer_postal_code', 'dealer_url', 'dealer_email', 'dealer_phone', 
                            'vehicle_id', 'vin', 'uvc', 'msrp', 'series', 'style', 'has_leather', 
                            'has_navigation', 'exterior_color', 'exterior_color_category', 
                            'interior_color', 'interior_color_category', 'price_analysis', 
                            'wheelbase_from_vin', 'drivetrain_from_vin', 'engine_from_vin', 
                            'price_history_delimited', 'distance_to_dealer', 'location_score', 
                            'listing_dropoff_date']
            self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            missing_values = self.df.isnull().sum().sum()
            self.logger.info(f"Total missing values before filling: {missing_values}")
            self.df.fillna(0, inplace=True)

            duplicates = self.df.duplicated().sum()
            if duplicates > 0:
                self.df.drop_duplicates(inplace=True)
                self.logger.info(f"Removed {duplicates} duplicate rows")

            self.df['transmission_from_vin'] = self.df['transmission_from_vin'].replace({'6': 'M', '7': 'A'})

            Q1, Q3 = self.df['price'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            initial_rows = len(self.df)
            self.df = self.df[(self.df['price'] >= lower_bound) & (self.df['price'] <= upper_bound)]
            self.logger.info(f"Removed {initial_rows - len(self.df)} outliers based on IQR.")

            scaler = MinMaxScaler()
            numerical_features = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'transmission_from_vin' in numerical_features:
                numerical_features.remove('transmission_from_vin')
            self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])

            self.logger.info("Data preprocessing completed successfully")
            return self.df
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def split_data(self, target_column, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.logger.info("Data split into training and testing sets")

    def encode_train_test(self, categorical_cols, method="label"):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        
        for col in categorical_cols:
            if col in X_train.columns:
                if method == "label":
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        X_train[col] = self.encoders[col].fit_transform(X_train[col])
                    X_test[col] = X_test[col].map(lambda x: self.encoders[col].transform([x])[0] if x in self.encoders[col].classes_ else -1)
                elif method == "onehot":
                    combined = pd.concat([X_train, X_test], axis=0)
                    combined = pd.get_dummies(combined, columns=[col], drop_first=True)
                    X_train = combined.iloc[:len(X_train), :]
                    X_test = combined.iloc[len(X_train):, :]
                else:
                    raise ValueError("Invalid encoding method. Choose 'onehot' or 'label'.")
        
        self.X_train = X_train
        self.X_test = X_test
        self.logger.info("Categorical features encoded")
        return self.X_train, self.X_test

    def save_processed_data(self, output_path="processed_data.csv"):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
        else:
            self.logger.error("Error: No data to save. Please preprocess the data first.")

if __name__ == "__main__":
    preprocessor = DataPreprocessor('data/raw/CBB_Listings.csv', logger)
    preprocessor.load_data()
    preprocessor.explore_data()
    processed_data = preprocessor.preprocess_data()
    
    categorical_cols = ['make', 'model', 'stock_type', 'dealer_type', 'fuel_type_from_vin']
    
    preprocessor.split_data(target_column='transmission_from_vin')
    preprocessor.encode_train_test(categorical_cols, method="label")
    preprocessor.save_processed_data('data/processed/processed_data.csv')