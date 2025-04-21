# model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import os

from data_preprocessing import load_data, preprocess_data

def train_decision_tree(X_train, y_train, preprocessor):
    """Train a Decision Tree classifier."""
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train the model
    clf.fit(X_train_processed, y_train)
    
    return clf

def train_knn(X_train, y_train, preprocessor):
    """Train a K-Nearest Neighbors classifier."""
    clf = KNeighborsClassifier(n_neighbors=5)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train the model
    clf.fit(X_train_processed, y_train)
    
    return clf

def evaluate_models(X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols):
    """Evaluate multiple models using cross-validation and ROC curves."""
    
    # Create directory for visualizations if it doesn't exist
    if not os.path.exists('../visualizations'):
        os.makedirs('../visualizations')
    
    # Train models
    dt_model = train_decision_tree(X_train, y_train, preprocessor)
    knn_model = train_knn(X_train, y_train, preprocessor)
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Make predictions
    dt_preds = dt_model.predict(X_test_processed)
    knn_preds = knn_model.predict(X_test_processed)
    
    # Print model evaluation
    print("\nDecision Tree Results:")
    print(classification_report(y_test, dt_preds))
    
    print("\nKNN Results:")
    print(classification_report(y_test, knn_preds))
    
    # Create ROC curves
    plt.figure(figsize=(10, 8))
    
    # Decision Tree ROC
    dt_probs = dt_model.predict_proba(X_test_processed)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    plt.plot(fpr_dt, tpr_dt, lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
    
    # KNN ROC
    knn_probs = knn_model.predict_proba(X_test_processed)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    plt.plot(fpr_knn, tpr_knn, lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
    
    # Random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('../visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze feature importance (Decision Tree)
    if hasattr(dt_model, 'feature_importances_'):
        # Get feature names after one-hot encoding
        feature_names = num_cols.copy()
        for col in cat_cols:
            for category in X_train[col].unique()[1:]:  # Skip first category due to drop='first'
                feature_names.append(f"{col}_{category}")
                
        # Create DataFrame for better visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(dt_model.feature_importances_)],
            'Importance': dt_model.feature_importances_
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
        plt.title('Top 10 Features by Importance (Decision Tree)', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return dt_model, knn_model, roc_auc_dt, roc_auc_knn

def run_model_training():
    """Run the entire model training and evaluation process."""
    
    # Load the data
    df = load_data('../data/Customer_Churn.xlsx')
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = preprocess_data(df)
    
    # Train and evaluate the models
    dt_model, knn_model, dt_auc, knn_auc = evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols)
    
    print("\nModel training and evaluation complete!")
    print(f"Decision Tree AUC: {dt_auc:.4f}")
    print(f"KNN AUC: {knn_auc:.4f}")
    print(f"Best model: {'Decision Tree' if dt_auc > knn_auc else 'KNN'}")

if __name__ == "__main__":
    run_model_training()
