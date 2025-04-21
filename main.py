# main.py
import os
import sys
import pandas as pd
import time

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from data_preprocessing import load_data, preprocess_data
from exploratory_analysis import run_exploratory_analysis
from model_training import run_model_training

def main():
    """Run the complete customer churn analysis pipeline."""
    start_time = time.time()
    
    print("="*80)
    print("CUSTOMER CHURN PREDICTION ANALYSIS")
    print("="*80)
    
    # 1. Data Preprocessing
    print("\nSTEP 1: DATA PREPROCESSING")
    print("-"*50)
    try:
        df = load_data('./data/Customer_Churn.xlsx')
        X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = preprocess_data(df)
        print("Data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return
    
    # 2. Exploratory Data Analysis
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
    print("-"*50)
    try:
        run_exploratory_analysis()
        print("Exploratory analysis completed successfully!")
    except Exception as e:
        print(f"Error in exploratory analysis: {e}")
    
    # 3. Model Training and Evaluation
    print("\nSTEP 3: MODEL TRAINING AND EVALUATION")
    print("-"*50)
    try:
        run_model_training()
        print("Model training and evaluation completed successfully!")
    except Exception as e:
        print(f"Error in model training: {e}")
    
    # Print execution time
    execution_time = time.time() - start_time
    print("\n"+"="*80)
    print(f"Analysis completed in {execution_time:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
