# Customer Churn Prediction Analysis

## Project Overview
This project analyzes customer churn in a telecommunications company using data mining techniques. The goal is to predict whether customers will leave (churn) or stay with the company based on various attributes.

## Data Description
The dataset contains the following features:
- COLLEGE: Whether the customer is college educated
- INCOME: Annual income
- OVERAGE: Average overcharges per month
- LEFTOVER: Average % leftover minutes per month
- HOUSE: Value of dwelling (from census tract)
- HANDSET_PRICE: Cost of phone
- OVER_15MINS_CALLS_PER_MONTH: Average number of long calls per month
- AVERAGE_CALL_DURATION: Average call duration
- REPORTED_SATISFACTION: Reported level of satisfaction
- REPORTED_USAGE_LEVEL: Self-reported usage level
- CONSIDERING_CHANGE_OF_PLAN: Was the customer considering changing plan?
- LEAVE: Target variable (whether customer left or stayed)

## Project Structure
- `data/`: Contains the dataset
- `code/`: Python scripts for data processing and analysis
- `visualizations/`: Generated charts and plots

## Methodology
This project employs multiple predictive models including Decision Trees and K-Nearest Neighbors to predict customer churn. The models are evaluated using cross-validation and ROC curves.
