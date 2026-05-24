
#missing value handling
#encoding
#scaling

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def handle_missing_values(df):

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    imputer = SimpleImputer(strategy='mean')

    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    return df


def encode_features(df):

    encoder = LabelEncoder()

    categorical_cols = ['Gender', 'Married']

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df