# exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_output_dir():
    """Create directory for visualizations if it doesn't exist."""
    if not os.path.exists('../visualizations'):
        os.makedirs('../visualizations')

def analyze_churn_distribution(df):
    """Visualize distribution of customer churn."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='LEAVE', data=df, palette=['#ff9999','#66b3ff'])
    plt.title('Distribution of Customer Churn', fontsize=16)
    plt.xlabel('Customer Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', 
                    fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2), 
                    ha = 'center', va = 'center', 
                    fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(df):
    """Visualize correlation between variables."""
    # Create a binary version of LEAVE for correlation
    df_binary = df.copy()
    df_binary['LEAVE_BINARY'] = (df_binary['LEAVE'] == 'LEAVE').astype(int)
    
    # Convert categorical variables to numeric for correlation analysis
    for col in df_binary.select_dtypes(include=['object']).columns:
        if col != 'LEAVE':  # Skip the original LEAVE column
            df_binary[col] = pd.factorize(df_binary[col])[0]
    
    # Calculate correlation matrix
    correlation_matrix = df_binary.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                          cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
                          square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_satisfaction_churn(df):
    """Analyze relationship between satisfaction and churn."""
    # Calculate the proportion of churned customers by satisfaction level
    churn_by_satisfaction = pd.crosstab(
        df['REPORTED_SATISFACTION'], 
        df['LEAVE'], 
        normalize='index'
    ).reset_index()
    
    # Create the chart
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot the bars
    ax = sns.barplot(
        x='REPORTED_SATISFACTION', 
        y='LEAVE', 
        data=churn_by_satisfaction,
        color='#ff6b6b'
    )
    
    # Customize the chart
    plt.title('Churn Rate by Customer Satisfaction Level', fontsize=16)
    plt.xlabel('Reported Satisfaction', fontsize=12)
    plt.ylabel('Proportion of Customers Who Left', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        percentage = p.get_height() * 100
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                    ha = 'center', va = 'bottom', 
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../visualizations/satisfaction_churn.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_exploratory_analysis():
    """Run all exploratory analysis functions."""
    create_output_dir()
    
    # Load the data
    df = pd.read_excel('../data/Customer_Churn.xlsx')
    
    # Run analysis
    analyze_churn_distribution(df)
    create_correlation_heatmap(df)
    analyze_satisfaction_churn(df)
    
    print("Exploratory analysis complete! Visualizations saved in '../visualizations' directory.")

if __name__ == "__main__":
    run_exploratory_analysis()
