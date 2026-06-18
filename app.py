import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Bank Churn Risk Calculator", layout="wide")
st.title("🏦 Bank Customer Churn Risk Calculator")
st.write("Predict churn probability and explore what-if scenarios")

# Sidebar inputs
st.sidebar.header("Customer Profile")
age = st.sidebar.slider("Age", 18, 92, 40)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
balance = st.sidebar.number_input("Balance (€)", 0.0, 250000.0, 50000.0)
salary = st.sidebar.number_input("Estimated Salary (€)", 0.0, 200000.0, 70000.0)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Build input row matching model columns
input_dict = {col: 0 for col in model_columns}
input_dict['CreditScore'] = credit_score
input_dict['Age'] = age
input_dict['Tenure'] = tenure
input_dict['Balance'] = balance
input_dict['NumOfProducts'] = num_products
input_dict['HasCrCard'] = 1 if has_card == "Yes" else 0
input_dict['IsActiveMember'] = 1 if is_active == "Yes" else 0
input_dict['EstimatedSalary'] = salary

if 'BalanceToSalary' in input_dict:
    input_dict['BalanceToSalary'] = balance / (salary + 1)
if 'ProductDensity' in input_dict:
    input_dict['ProductDensity'] = num_products / (tenure + 1)
if 'EngagementProduct' in input_dict:
    input_dict['EngagementProduct'] = (1 if is_active=="Yes" else 0) * num_products
if 'AgeByTenure' in input_dict:
    input_dict['AgeByTenure'] = age * tenure

geo_col = f'Geography_{geography}'
if geo_col in input_dict:
    input_dict[geo_col] = 1
gender_col = f'Gender_{gender}'
if gender_col in input_dict:
    input_dict[gender_col] = 1

input_df = pd.DataFrame([input_dict])[model_columns]

# Predict
prob = model.predict_proba(input_df)[0][1] * 100

# Display result
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{prob:.1f}%")
    if prob < 30:
        st.success("Low Risk — Customer likely to stay")
    elif prob < 60:
        st.warning("Medium Risk — Monitor and engage")
    else:
        st.error("High Risk — Immediate retention action needed")

with col2:
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie([prob, 100-prob], labels=['Churn Risk','Retention'], 
           colors=['#e74c3c','#27ae60'], autopct='%1.1f%%')
    st.pyplot(fig)

# Feature importance
st.subheader("📊 What Drives Churn? (Model Feature Importance)")
importances = pd.Series(model.feature_importances_, index=model_columns)
importances = importances.sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(8,5))
importances.plot(kind='barh', ax=ax2, color='steelblue')
ax2.invert_yaxis()
st.pyplot(fig2)
