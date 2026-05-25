import sqlite3


def create_connection():

    connection = sqlite3.connect(
        "database/loan_app.db"
    )

    return connection


def create_table():

    connection = create_connection()

    cursor = connection.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            Gender TEXT,

            Married TEXT,

            Dependents TEXT,

            Education TEXT,

            Self_Employed TEXT,

            ApplicantIncome REAL,

            CoapplicantIncome REAL,

            LoanAmount REAL,

            Loan_Amount_Term REAL,

            Credit_History REAL,

            Prediction TEXT

        )

    """)

    connection.commit()

    connection.close()


def save_prediction(data, prediction):

    connection = create_connection()

    cursor = connection.cursor()

    cursor.execute("""

        INSERT INTO predictions (

            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Prediction

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        data["Gender"],

        data["Married"],

        data["Dependents"],

        data["Education"],

        data["Self_Employed"],

        data["ApplicantIncome"],

        data["CoapplicantIncome"],

        data["LoanAmount"],

        data["Loan_Amount_Term"],

        data["Credit_History"],

        prediction

    ))

    connection.commit()

    connection.close()