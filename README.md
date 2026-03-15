#Credit Risk Prediction System
Project Overview

This project predicts whether a loan applicant is likely to default on a loan based on financial and personal information.
The goal is to help financial institutions assess credit risk before approving loans.

Dataset

The project uses three datasets:

Customers – age, gender, marital status, employment status, income, dependants

Loans – loan type, loan amount, sanction amount, processing fee, loan duration

Bureau Data – bank balance, open accounts, closed accounts

These datasets are merged using a customer ID.

Feature Engineering

Additional features were created to improve prediction:

Total Available Funds

Loan to Funds Ratio

Debt Ratio

Open to Closed Account Ratio

Balance Minus Loan

Model Used

The model used is LightGBM Classifier, which is efficient and performs well on structured datasets.

Model Evaluation

The model was evaluated using:

Accuracy Score

Precision and Recall

Confusion Matrix

Technologies Used

Python

Pandas

NumPy

Scikit-learn

LightGBM

Streamlit

Matplotlib

Seaborn
