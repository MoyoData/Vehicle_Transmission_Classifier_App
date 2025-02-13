# src/utils/helpers.py

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        return pd.read_csv(self.filepath, on_bad_lines='skip')

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df.fillna('unknown', inplace=True)
        self.df = self.df[~self.df.duplicated()]
        return self.df

    def preprocess_data(self):
        categorical_cols = ['make', 'model', 'stock_type', 'dealer_type', 'fuel_type_from_vin']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
        return self.df