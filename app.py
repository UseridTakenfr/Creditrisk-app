import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Credit Risk Prediction App")
st.title("🔍 Credit Risk Prediction App")

# Load data
@st.cache_data
def load_data():
    customers = pd.read_csv("customers.csv")
    loans = pd.read_csv("loans.csv")
    bureau = pd.read_csv("bureau_data.csv")
    return customers, loans, bureau

customers, loans, bureau = load_data()

# Merge datasets
df = customers.merge(loans, on="cust_id").merge(bureau, on="cust_id")

# Feature engineering
df["loan_to_income_ratio"] = df["loan_amount"] / df["income"]
df["loan_utilization"] = df["loan_amount"] / df["sanction_amount"]
df["account_activity"] = df["number_of_open_accounts"] + df["number_of_closed_accounts"]

# Risk classification
def classify_risk(row):
    score = 0
    if row["loan_to_income_ratio"] > 2: score += 2
    elif row["loan_to_income_ratio"] > 1: score += 1
    if row["account_activity"] < 2: score += 2
    elif row["account_activity"] < 4: score += 1
    if row["loan_utilization"] > 0.9: score += 1
    if row["total_loan_months"] > 100: score -= 1

    if score >= 4: return "Poor"
    elif score >= 2: return "Average"
    else: return "Good"

df["credit_risk"] = df.apply(classify_risk, axis=1)

# Encode categorical columns
label_cols = ["gender", "marital_status", "employment_status", "loan_purpose", "loan_type"]
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and labels
features = [
    "age", "gender", "marital_status", "employment_status", "income",
    "number_of_dependants", "loan_purpose", "loan_type",
    "sanction_amount", "loan_amount", "processing_fee",
    "number_of_open_accounts", "number_of_closed_accounts", "total_loan_months",
    "loan_to_income_ratio", "loan_utilization", "account_activity"
]
X = df[features]
y = df["credit_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric("📊 Model Accuracy", f"{accuracy:.2%}")

# Optional: detailed report
with st.expander("📈 Show Classification Report"):
    st.text(classification_report(y_test, y_pred))

# Prediction UI
st.header("📥 Predict Credit Risk for a New Applicant")

with st.form("form"):
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    marital = st.selectbox("Marital Status", encoders["marital_status"].classes_)
    employment = st.selectbox("Employment Status", encoders["employment_status"].classes_)
    income = st.number_input("Income", 0, 10000000, 500000)
    dependants = st.slider("Number of Dependants", 0, 10, 0)
    loan_purpose = st.selectbox("Loan Purpose", encoders["loan_purpose"].classes_)
    loan_type = st.selectbox("Loan Type", encoders["loan_type"].classes_)
    sanction_amount = st.number_input("Sanction Amount", 0, 10000000, 500000)
    loan_amount = st.number_input("Loan Amount", 0, 10000000, 400000)
    processing_fee = st.number_input("Processing Fee", 0, 100000, 1000)
    open_acc = st.slider("Open Accounts", 0, 10, 1)
    closed_acc = st.slider("Closed Accounts", 0, 10, 1)
    total_months = st.number_input("Total Loan Months", 1, 500, 60)

    submit = st.form_submit_button("Predict")

    if submit:
        ltir = loan_amount / income if income != 0 else 0
        utilization = loan_amount / sanction_amount if sanction_amount != 0 else 0
        activity = open_acc + closed_acc

        input_data = pd.DataFrame([[
            age,
            encoders["gender"].transform([gender])[0],
            encoders["marital_status"].transform([marital])[0],
            encoders["employment_status"].transform([employment])[0],
            income, dependants,
            encoders["loan_purpose"].transform([loan_purpose])[0],
            encoders["loan_type"].transform([loan_type])[0],
            sanction_amount, loan_amount, processing_fee,
            open_acc, closed_acc, total_months,
            ltir, utilization, activity
        ]], columns=features)

        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Credit Risk: *{prediction}*")
