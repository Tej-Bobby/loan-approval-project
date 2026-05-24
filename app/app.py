import streamlit as st
import joblib
import numpy as np


model = joblib.load("../models/loan_model.pkl")

st.title("Loan Approval Prediction")

income = st.number_input("Applicant Income")

loan_amount = st.number_input("Loan Amount")


if st.button("Predict"):

    data = np.array([[income, loan_amount]])

    prediction = model.predict(data)

    st.write(prediction[0])