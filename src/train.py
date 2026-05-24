import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import handle_missing_values
from data_preprocessing import encode_features

from feature_engineering import log_transform


df = pd.read_csv("data/raw/loan.csv")

df = handle_missing_values(df)

df = encode_features(df)

df = log_transform(df, ['ApplicantIncome'])

X = df.drop('Loan_Status', axis=1)

y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Training completed")

import joblib

joblib.dump(model, "models/loan_model.pkl")