import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.encoders = {}  # Store fitted encoders for consistency

    def load_data(self):
        self.df = pd.read_csv(self.file_path, on_bad_lines='skip')
        return self.df

    def explore_data(self):
        print(self.df.head())
        print(self.df.tail())
        print(self.df.columns)
        print(self.df.info())
        print(self.df.shape)
        print(self.df.dtypes)
        print(self.df.nunique())
        print(self.df.duplicated().sum())

    def preprocess_data(self):
        # Drop unnecessary columns
        self.df = self.df.drop(columns=['dealer_email', 'listing_id', 'listing_heading', 'listing_type', 'listing_url',
                                        'listing_first_date', 'days_on_market', 'dealer_id', 'dealer_name',
                                        'dealer_street', 'dealer_city', 'dealer_province', 'dealer_postal_code',
                                        'dealer_url', 'dealer_email', 'dealer_phone', 'vehicle_id', 'vin', 'uvc', 'msrp', 'series', 'style',
                                        'has_leather', 'has_navigation', 'exterior_color',
                                        'exterior_color_category', 'interior_color', 'interior_color_category',
                                        'price_analysis', 'wheelbase_from_vin', 'drivetrain_from_vin',
                                        'engine_from_vin', 'price_history_delimited', 'distance_to_dealer',
                                        'location_score', 'listing_dropoff_date'])

        # Handle missing values
        if self.df.empty:
            print("Warning: DataFrame is empty before fillna()")
        print(self.df.isnull().sum())
        self.df = self.df.fillna(0)

        # Checking for duplicate rows
        duplicate_rows = self.df.duplicated()
        print("Duplicate Rows:", duplicate_rows.sum())

        # Remove duplicate rows
        self.df = self.df.drop_duplicates()

        def clean_transmission_column(df, column):
            # Replace '6' with 'M' and '7' with 'A'
            df[column] = df[column].replace({'6': 'M', '7': 'A'})
            return df
        self.df = clean_transmission_column(self.df, 'transmission_from_vin')

        # Remove outliers in the 'price' column using IQR
        Q1 = self.df['price'].quantile(0.25)
        Q3 = self.df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df['price'] >= lower_bound) & (self.df['price'] <= upper_bound)]
        print("Outliers removed based on IQR method.")

        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_features = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])

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
    
    
    def encode_train_test(self, categorical_cols, method="label"):
        """
        Encodes categorical columns in both training and test datasets, ensuring consistency.

        :param X_train: Training DataFrame
        :param X_test: Test DataFrame
        :param categorical_cols: List of categorical column names
        :param method: Encoding method - "label" or "onehot"
        :return: Encoded X_train and X_test DataFrames
        """
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
        return self.X_train, self.X_test

    def save_processed_data(self, output_path="processed_data.csv"):
        """Saves the processed data to a CSV file."""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("Error: No data to save. Please preprocess the data first.")


if __name__ == "__main__":
    preprocessor = DataPreprocessor('data/raw/CBB_Listings.csv')
    preprocessor.load_data()
    preprocessor.explore_data()
    processed_data = preprocessor.preprocess_data()

    # Define categorical columns
    categorical_cols = ['make', 'model', 'stock_type', 'dealer_type', 'fuel_type_from_vin']

    # Split the data
    preprocessor.split_data(target_column='transmission_from_vin')

    # Encode categorical features
    preprocessor.encode_train_test(categorical_cols, method="label")

    # Save the processed data
    preprocessor.save_processed_data('processed_data.csv')