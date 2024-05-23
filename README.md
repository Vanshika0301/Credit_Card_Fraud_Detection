# Credit Card Fraud Detection

![image](https://github.com/Vanshika0301/Credit_Card_Fraud_Detection/assets/146732449/73471744-daaf-4942-b975-136d3e3a3eee)

## Introduction
This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The aim is to help credit card companies identify and prevent fraudulent activities, thereby protecting customers from unauthorized charges.

## Business Objective
The primary goal is to analyze credit card transaction data and accurately identify fraudulent transactions. By doing so, we aim to minimize financial losses for both the credit card companies and their customers.

## Dataset
The dataset used in this project is obtained from Kaggle and contains credit card transactions made by European cardholders over two days in September 2013. It comprises 284,807 transactions, out of which only 492 are classified as frauds.

## Methodology
- Data Preparation/Preprocessing: Cleaning the dataset, handling missing values, outliers, and unnecessary features.
- Exploratory Data Analysis (EDA): Analyzing the distribution of features, identifying patterns, and visualizing relationships between variables.
- Model Building: Employing various classification algorithms to train models for fraud detection.
- Hyper-parameter Optimization: Fine-tuning the parameters of the best-performing model using RandomizedSearchCV.
- Evaluation: Evaluating the models based on performance metrics such as precision, recall, and F1-score.
- Saving the Model: Saving the trained model and feature columns for future use.

## Model Performance
Random Forest Classifier:
- Precision: 88%
- Recall: 80%
- F1-score: 84%

## Files Included
- credit_card_fraud_detection.ipynb: Jupyter notebook containing the entire codebase, including data preprocessing, model building, and evaluation.
- credit_card_fraud_detection.pickle: Pickle file containing the trained Random Forest Classifier model.
- columns.json: JSON file containing the feature columns used in the model.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, imbalanced-learn

## Usage
- Clone the repository.
- Install the required dependencies.
- Run the credit_card_fraud_detection.ipynb notebook to train and evaluate the model.
= Use the saved model (credit_card_fraud_detection.pickle) for fraud detection in real-time applications.

## Future Improvements
- Explore advanced feature engineering techniques.
- Experiment with ensemble methods and deep learning algorithms.
- Deploy the model as a web service for real-time fraud detection.
