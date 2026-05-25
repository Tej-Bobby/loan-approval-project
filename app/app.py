import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import pandas as pd
import sqlite3

from src.predict import (
    load_model,
    load_preprocessor,
    make_prediction
)

from src.database import (
    create_table,
    save_prediction
)


# Create Database Table
create_table()


# Load Model
model = load_model("models/model.pkl")


# Load Preprocessor
preprocessor = load_preprocessor(
    "models/preprocessor.pkl"
)


# App Title
st.title("Loan Approval Prediction System")


# User Inputs
Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

Education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

Self_Employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

ApplicantIncome = st.number_input(
    "Applicant Income",
    min_value=0
)

CoapplicantIncome = st.number_input(
    "Coapplicant Income",
    min_value=0
)

LoanAmount = st.number_input(
    "Loan Amount",
    min_value=0
)

Loan_Amount_Term = st.number_input(
    "Loan Amount Term",
    min_value=0
)

Credit_History = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)


# Predict Button
if st.button("Predict"):

    # Feature Engineering
    Has_Coapplicant = int(
        CoapplicantIncome > 0
    )

    # Input Data
    input_data = {

        "Gender": Gender,

        "Married": Married,

        "Dependents": Dependents,

        "Education": Education,

        "Self_Employed": Self_Employed,

        "ApplicantIncome": ApplicantIncome,

        "CoapplicantIncome": CoapplicantIncome,

        "LoanAmount": LoanAmount,

        "Loan_Amount_Term": Loan_Amount_Term,

        "Credit_History": Credit_History,

        "Has_Coapplicant": Has_Coapplicant
    }

    # Prediction
    prediction = make_prediction(
        input_data,
        model,
        preprocessor
    )

    # Convert Output
    if prediction == "Y":

        result = "Approved"

        st.success("Loan Approved")

    else:

        result = "Rejected"

        st.error("Loan Rejected")

    # Save Prediction to Database
    save_prediction(
        input_data,
        result
    )


# Show Prediction History
if st.button("Show Prediction History"):

    connection = sqlite3.connect(
        "database/loan_app.db"
    )

    query = """
        SELECT * FROM predictions
    """

    history_df = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    st.dataframe(history_df)