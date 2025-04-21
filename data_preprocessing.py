# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the customer churn dataset."""
    df = pd.read_excel(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Preprocess the data for modeling."""
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('LEAVE', axis=1)
    y = (df['LEAVE'] == 'LEAVE').astype(int)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols

if __name__ == "__main__":
    # Test the functions
    df = load_data("../data/Customer_Churn.xlsx")
    X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = preprocess_data(df)
    print("Data preprocessing complete!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
