#this will be where the CI/CB pipeline will be built for taking in new data and predicting if a customer will churn or not.

#importing the libraries
import joblib
import streamlit as st
import sklearn
import numpy as np
import pandas as pd

# Load the model from the file
loaded_model = joblib.load('random_forest_model.pkl')

#input from user for getting output from the model:
credit = st.number_input("Credit Score", min_value=0, max_value=850, value=720)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.slider("Tenure (years)", 0, 10, 2)
balance = st.number_input("Balance", min_value=0, value=1000)
salary = st.number_input("Salary", min_value=0,  value=50000)


# Assuming test_array is already defined
test_array = [credit, age, tenure, balance, salary]  # Example values

# Define the column names
column_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

# Create a DataFrame
df_test = pd.DataFrame([test_array], columns=column_names)

X_test = df_test



# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)

# Evaluate the loaded model
# print("Accuracy of loaded model:", accuracy_score(y_test, predictions))

if st.button("Predict Churn"):
    prediction = loaded_model.predict(X_test)
    st.write("Churn Prediction:", "Churn" if prediction[0] == 1 else "No Churn")

#run script: streamlit run working_model.py