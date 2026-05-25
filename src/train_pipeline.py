import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    handle_missing_values,
    handle_outliers_and_skewness,
    create_preprocessor
)

from src.feature_engineering import create_features

from src.train import (
    get_models,
    cross_validate_models,
    get_best_model,
    perform_grid_search
)

from src.evaluate import evaluate_model

from src.utils import save_object


def run_training_pipeline():

    # Load Dataset
    df = pd.read_csv("data/raw/Loan.csv")

    # Separate Features and Target
    X = df.drop("Loan_Status", axis=1)

    y = df["Loan_Status"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Handle Missing Values
    X_train, X_test = handle_missing_values(
        X_train,
        X_test
    )

    # Feature Engineering
    X_train, X_test = create_features(
        X_train,
        X_test
    )

    # Handle Outliers and Skewness
    X_train, X_test = handle_outliers_and_skewness(
        X_train,
        X_test
    )

    # Create Preprocessor
    preprocessor = create_preprocessor()

    # Transform Data
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    # Get Models
    models = get_models()

    # Cross Validation
    results = cross_validate_models(
        models,
        X_train_transformed,
        y_train
    )

    # Best Model
    best_model_name, best_accuracy = get_best_model(
        results
    )

    # Perform Grid Search
    best_model = perform_grid_search(
        X_train_transformed,
        y_train
    )

    # Evaluate Model
    accuracy, report = evaluate_model(
        best_model,
        X_test_transformed,
        y_test
    )

    # Create Models Folder
    os.makedirs("models", exist_ok=True)

    # Save Model
    save_object(
        "models/model.pkl",
        best_model
    )

    # Save Preprocessor
    save_object(
        "models/preprocessor.pkl",
        preprocessor
    )

    print("\nTraining Pipeline Completed Successfully")


if __name__ == "__main__":

    run_training_pipeline()