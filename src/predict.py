import pickle
import pandas as pd


def load_model(model_path):

    with open(model_path, "rb") as file:

        model = pickle.load(file)

    return model


def load_preprocessor(preprocessor_path):

    with open(preprocessor_path, "rb") as file:

        preprocessor = pickle.load(file)

    return preprocessor


def make_prediction(input_data, model, preprocessor):

    # Convert Input Data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply Preprocessing
    transformed_data = preprocessor.transform(input_df)

    # Make Prediction
    prediction = model.predict(transformed_data)

    return prediction[0]