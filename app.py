import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

# Streamlit page config
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("🔍 Credit Risk Classifier")

# Load and merge datasets
@st.cache_data
def load_data():
    customers = pd.read_csv("customers.csv")
    loans = pd.read_csv("loans.csv")
    bureau = pd.read_csv("bureau_data.csv")
    df = customers.merge(loans, on="cust_id").merge(bureau, on="cust_id")
    return df

df = load_data()

# --- Preprocess ---
df.drop_duplicates(inplace=True)
df['age'] = df['age'].astype(str).str.extract(r'(\d+)').astype(int)

# Simulate bank balance if missing
if 'bank_balance_at_application' not in df.columns:
    df['bank_balance_at_application'] = df['income'] * 0.1

fields = [
    'age', 'gender', 'marital_status', 'employment_status',
    'income', 'number_of_dependants',
    'loan_type', 'sanction_amount', 'loan_amount', 'processing_fee',
    'bank_balance_at_application', 'number_of_open_accounts',
    'number_of_closed_accounts', 'total_loan_months'
]

fields.append('default')
df = df[fields[:-1]].dropna()

# Add features
df['total_available_funds'] = df['income'] + df['bank_balance_at_application']
df['loan_to_funds_ratio'] = np.log1p(df['loan_amount'] / (df['total_available_funds'] + 1))
df['balance_to_sanction_ratio'] = np.log1p(df['bank_balance_at_application'] / (df['sanction_amount'] + 1))
df['open_to_closed_ratio'] = np.log1p(df['number_of_open_accounts'] / (df['number_of_closed_accounts'] + 1))
df['balance_minus_loan'] = df['bank_balance_at_application'] - df['loan_amount']
df['debt_ratio'] = (df['loan_amount'] + df['processing_fee']) / (df['income'] + df['bank_balance_at_application'] + 1)

# Realistic and relaxed default label logic
df['default'] = (
    (df['loan_amount'] > (df['income'] * 12 + df['bank_balance_at_application'])) |
    (df['loan_to_funds_ratio'] > 1.5) |
    ((df['number_of_open_accounts'] > 10) & (df['open_to_closed_ratio'] > 3))
).astype(int)

# Balance the dataset
majority = df[df['default'] == 0]
minority = df[df['default'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])

# Encode categorical variables
encoders = {}
for col in ['gender', 'marital_status', 'employment_status', 'loan_type']:
    enc = LabelEncoder()
    df_balanced[col] = enc.fit_transform(df_balanced[col])
    encoders[col] = enc

# Split features and target
X = df_balanced.drop("default", axis=1)
y = df_balanced["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train model
model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.03,
    subsample=0.95,
    colsample_bytree=0.95,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

st.subheader("✅ Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
st.text(classification_report(y_test, preds))

# Smaller confusion matrix
fig, ax = plt.subplots(figsize=(1.5, 1.5))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig, use_container_width=False)

# Feature importance
importances = model.feature_importances_
feat_names = X_train.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})

st.subheader("📊 Feature Importance")
st.dataframe(feat_df.sort_values('Importance', ascending=False))

# --- Input for prediction ---
st.subheader("📥 Predict Credit Risk")

user_input = {}
valid_input = True

for col in ['gender', 'marital_status', 'employment_status', 'loan_type']:
    options = encoders[col].classes_
    value = st.selectbox(f"{col.replace('_', ' ').title()}", options)
    user_input[col] = encoders[col].transform([value])[0]

for col in [
    'age', 'income', 'number_of_dependants',
    'sanction_amount', 'loan_amount', 'processing_fee',
    'bank_balance_at_application', 'number_of_open_accounts',
    'number_of_closed_accounts', 'total_loan_months'
]:
    value = st.number_input(col.replace('_', ' ').title(), min_value=0, step=1)
    user_input[col] = value
    if col not in ['income', 'number_of_closed_accounts'] and value == 0:
        valid_input = False

user_input['total_available_funds'] = user_input['income'] + user_input['bank_balance_at_application']
user_input['loan_to_funds_ratio'] = np.log1p(user_input['loan_amount'] / (user_input['total_available_funds'] + 1))
user_input['balance_to_sanction_ratio'] = np.log1p(user_input['bank_balance_at_application'] / (user_input['sanction_amount'] + 1))
user_input['open_to_closed_ratio'] = np.log1p(user_input['number_of_open_accounts'] / (user_input['number_of_closed_accounts'] + 1))
user_input['balance_minus_loan'] = user_input['bank_balance_at_application'] - user_input['loan_amount']
user_input['debt_ratio'] = (user_input['loan_amount'] + user_input['processing_fee']) / (user_input['income'] + user_input['bank_balance_at_application'] + 1)

user_df = pd.DataFrame([user_input])
user_df = user_df[X_train.columns]

if st.button("🔮 Predict Risk"):
    if not valid_input or user_df.isnull().values.any():
        st.warning("⚠ Please enter non-zero values (except for income and number of closed accounts).")
    else:
        probability = model.predict_proba(user_df)[0][1] * 100

        # Updated relaxed label thresholds
        if probability < 30 and user_input['balance_minus_loan'] > 0:
            label = "🟢 Good"
        elif probability < 60:
            label = "🟡 Average"
        else:
            label = "🔴 Poor"

        st.success(f"Prediction: *{label}* ({probability:.2f}% chance of default)")
   
