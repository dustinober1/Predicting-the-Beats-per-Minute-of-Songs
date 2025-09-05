def load_data(file_path):
    import pandas as pd
    
    """Load the dataset from a specified CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by handling missing values and normalizing features."""
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())  # Simple imputation with mean
    
    # Normalize features (example: Min-Max scaling)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def split_data(df, target_column):
    """Split the dataset into features and target variable."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)