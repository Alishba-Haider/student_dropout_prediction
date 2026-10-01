# Student Dropout Prediction Using Machine Learning

## Project Overview

This project develops a machine learning system for predicting student dropout risk using Logistic Regression. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a Streamlit-based prediction application.

## Problem Statement

Student dropout is an important educational challenge. Identifying students who may be at higher risk can help educational institutions provide timely academic support and intervention.

## Machine Learning Workflow

Dataset
→ Data Cleaning
→ Exploratory Data Analysis
→ Data Preprocessing
→ Train-Test Split
→ Logistic Regression
→ Model Evaluation
→ Prediction
→ Streamlit Application

## Dataset

The project uses a cleaned student dropout dataset.

The target variable is:

`Dropout`

## Data Preprocessing

The following preprocessing steps were performed:

- Checked missing values
- Checked duplicate records
- Inspected dataset information and statistics
- Separated features and target variable
- Applied one-hot encoding to categorical features
- Applied StandardScaler to numerical feature representation

## Model

The selected machine learning algorithm is:

**Logistic Regression**

The dataset was divided into:

- 80% training data
- 20% testing data

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC-AUC

## Prediction Application

A Streamlit application was developed to allow users to enter student information and receive:

- Dropout probability
- Dropout risk category

## Educational Early-Warning System

The application can potentially be used as an educational early-warning tool. A predicted risk level can serve as an indicator for identifying students who may benefit from additional academic support.

The prediction should be treated as a support signal rather than a final decision about a student's academic future.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## How to Run

Install the required packages:

`pip install -r requirements.txt`

Run the application:

`python -m streamlit run app.py`

## Project Structure

```text
student-dropout-prediction/
│
├── app.py
├── student_dropout_cleaned.csv
├── dropout_model.pkl
├── dropout_scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── student_dropout.ipynb
