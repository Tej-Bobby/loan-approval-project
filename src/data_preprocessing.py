import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def handle_missing_values(X_train, X_test):

    numerical_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount"
    ]

    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Credit_History",
        "Loan_Amount_Term"
    ]

    # Handle Numerical Missing Values
    for col in numerical_cols:

        median_value = X_train[col].median()

        X_train[col] = X_train[col].fillna(median_value)

        X_test[col] = X_test[col].fillna(median_value)

    # Handle Categorical Missing Values
    for col in categorical_cols:

        mode_value = X_train[col].mode()[0]

        X_train[col] = X_train[col].fillna(mode_value)

        X_test[col] = X_test[col].fillna(mode_value)

    return X_train, X_test


def handle_outliers_and_skewness(X_train, X_test):

    continuous_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount"
    ]

    for col in continuous_cols:

        Q1 = X_train[col].quantile(0.25)

        Q3 = X_train[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        # Handle Outliers
        X_train[col] = X_train[col].clip(lower, upper)

        X_test[col] = X_test[col].clip(lower, upper)

        # Handle Skewness
        X_train[col] = np.log1p(X_train[col])

        X_test[col] = np.log1p(X_test[col])

    return X_train, X_test


def create_preprocessor():

    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed"
    ]

    numerical_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Has_Coapplicant"
    ]

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                StandardScaler(),
                numerical_columns
            ),

            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )

        ]
    )

    return preprocessor